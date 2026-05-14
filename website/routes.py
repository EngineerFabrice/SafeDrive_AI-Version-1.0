# website/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
from . import User, get_connection, get_user_by_id, bcrypt
from .yolo_detector import detect_person
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import threading
import os
import math
import smtplib
import json
from email.mime.text import MIMEText
from werkzeug.utils import secure_filename
from urllib.parse import quote

routes = Blueprint("routes", __name__)

# -------------------- MODEL PLACEHOLDERS --------------------
driver_model = None
drunk_model = None
model_lock = threading.Lock()  # To avoid race conditions

def load_models():
    """Load TensorFlow models lazily."""
    global driver_model, drunk_model
    with model_lock:
        if driver_model is None:
            driver_model = tf.keras.models.load_model("driver_alcoholism_model.h5")
        if drunk_model is None:
            drunk_model = tf.keras.models.load_model("Drunking_Detection_Model.h5")

CLASS_NAMES_DRIVER = ["Alcoholic", "Non-Alcoholic"]
CLASS_NAMES_DRUNK = ["Drunk", "Sober"]

# -------------------- IMAGE PREPROCESSING --------------------
def preprocess_image(img, target_size=(224,224)):
    """Resize, normalize, expand dims for model prediction."""
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------- Helpers --------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Return approximate distance in kilometers between two lat/lng points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def send_email_notification(recipient, subject, body):
    smtp_host = os.getenv('SMTP_HOST')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_user = os.getenv('SMTP_USER')
    smtp_pass = os.getenv('SMTP_PASS')

    if not smtp_host or not recipient:
        return False

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_user or 'no-reply@safedrive.ai'
    msg['To'] = recipient

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as exc:
        print(f"[email] Failed to send to {recipient}: {exc}")
        return False


def save_notification(user_id, title, message, channel='web'):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO notifications (user_id,channel,title,message,read_status,created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (user_id, channel, title, message, False, datetime.now())
        )
    except Exception as exc:
        if 'Unknown column' in str(exc) and 'channel' in str(exc):
            cursor.execute(
                "INSERT INTO notifications (user_id,title,message,read_status,created_at) "
                "VALUES (%s,%s,%s,%s,%s)",
                (user_id, title, message, False, datetime.now())
            )
        else:
            cursor.close()
            conn.close()
            raise
    conn.commit()
    cursor.close()
    conn.close()

    if channel == 'email':
        user = get_user_by_id(user_id)
        if user:
            send_email_notification(user.email, title, message)


MESSAGE_COL_MAP = None

def get_message_column_names():
    global MESSAGE_COL_MAP
    if MESSAGE_COL_MAP is not None:
        return MESSAGE_COL_MAP

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SHOW COLUMNS FROM messages")
        cols = {row['Field'] for row in cursor.fetchall()}
    except Exception:
        cols = set()
    cursor.close()
    conn.close()

    if {'body', 'status', 'created_at'}.issubset(cols):
        MESSAGE_COL_MAP = {
            'body': 'body',
            'status': 'status',
            'created_at': 'created_at',
            'subject': 'subject' if 'subject' in cols else None
        }
    elif {'content', 'read_status', 'sent_at'}.issubset(cols):
        MESSAGE_COL_MAP = {
            'body': 'content',
            'status': 'read_status',
            'created_at': 'sent_at',
            'subject': 'subject' if 'subject' in cols else None
        }
    else:
        MESSAGE_COL_MAP = {
            'body': 'body',
            'status': 'status',
            'created_at': 'created_at',
            'subject': 'subject' if 'subject' in cols else None
        }
    return MESSAGE_COL_MAP


def send_notification(user_id, title, message, channels=None):
    if channels is None:
        channels = ['web', 'email']
    for channel in channels:
        save_notification(user_id, title, message, channel)


def find_nearest_available_driver(pickup_lat, pickup_lng):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT dl.driver_id, dl.latitude, dl.longitude, dl.available, u.username, u.email "
        "FROM driver_locations dl "
        "JOIN users u ON dl.driver_id = u.id "
        "WHERE dl.available=1 AND u.role='driver'"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    nearest = None
    best_distance = None
    for row in rows:
        try:
            dist = haversine_distance(pickup_lat, pickup_lng, float(row['latitude']), float(row['longitude']))
        except Exception:
            continue
        if best_distance is None or dist < best_distance:
            best_distance = dist
            nearest = row
    return nearest, best_distance


def assign_driver_to_trip(user_id, pickup_address, dropoff_address, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
    assigned_driver, distance = find_nearest_available_driver(pickup_lat, pickup_lng)
    conn = get_connection()
    cursor = conn.cursor()
    if assigned_driver:
        cursor.execute(
            "INSERT INTO trips (user_id, driver_id, pickup_address, pickup_lat, pickup_lng, dropoff_address, dropoff_lat, dropoff_lng, status, requested_at, assigned_at, distance_km) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (user_id, assigned_driver['driver_id'], pickup_address, pickup_lat, pickup_lng, dropoff_address, dropoff_lat, dropoff_lng, 'assigned', datetime.now(), datetime.now(), round(distance or 0, 2))
        )
        trip_id = cursor.lastrowid
        cursor.execute("UPDATE driver_locations SET available=0, updated_at=%s WHERE driver_id=%s", (datetime.now(), assigned_driver['driver_id']))
        conn.commit()
        cursor.close()
        conn.close()
        send_notification(assigned_driver['driver_id'], 'New Ride Assigned', f'A new ride has been assigned near {pickup_address}.', channels=['web', 'email'])
        return trip_id, assigned_driver

    cursor.execute(
        "INSERT INTO trips (user_id, pickup_address, pickup_lat, pickup_lng, dropoff_address, dropoff_lat, dropoff_lng, status, requested_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (user_id, pickup_address, pickup_lat, pickup_lng, dropoff_address, dropoff_lat, dropoff_lng, 'pending', datetime.now())
    )
    trip_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    return trip_id, None

# ========================= HOME =========================
@routes.route('/')
def home():
    return render_template('home.html')

# ========================= REGISTER =========================
@routes.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form.get('role', 'passenger')

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            flash('❌ Email already exists.', 'danger')
            cursor.close()
            conn.close()
            return redirect(url_for('routes.register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        cursor.execute(
            "INSERT INTO users (username,email,password,role) VALUES (%s,%s,%s,%s)",
            (username,email,hashed_password,role)
        )
        conn.commit()
        cursor.close()
        conn.close()
        flash('✅ Registration successful! Please login.', 'success')
        return redirect(url_for('routes.login'))
    return render_template('register.html')

# ========================= LOGIN =========================
@routes.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, password, role FROM users WHERE email=%s", (email,))
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()

        if user_data and bcrypt.check_password_hash(user_data['password'], password):
            user = User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password=user_data['password'],
                role=user_data['role']
            )
            login_user(user)
            flash(f'Welcome {user.username}!', 'success')
            if user.is_admin():
                return redirect(url_for('routes.admin_dashboard'))
            elif user.is_chef():
                return redirect(url_for('routes.chef_dashboard'))
            elif user.is_passenger():
                return redirect(url_for('routes.passenger_dashboard'))
            else:
                return redirect(url_for('routes.driver_dashboard'))

        flash('❌ Incorrect email or password.', 'danger')
    return render_template('login.html')

# ========================= LOGOUT =========================
@routes.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('routes.home'))

# ========================= ADMIN DASHBOARD =========================
@routes.route('/admin-dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin():
        flash("⚠️ Access denied.", "danger")
        return redirect(url_for("routes.home"))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, role FROM users")
    users = cursor.fetchall()

    total_users = len(users)
    total_drivers = sum(1 for u in users if u['role'] == 'driver')
    total_chefs = sum(1 for u in users if u['role'] == 'chef')
    total_admins = sum(1 for u in users if u['role'] == 'admin')

    cursor.execute(
        "SELECT u.username AS driver_name, r.detection_type, r.status, r.timestamp "
        "FROM driver_detection_reports r "
        "LEFT JOIN users u ON r.driver_id=u.id "
        "ORDER BY r.timestamp DESC LIMIT 20"
    )
    reports = cursor.fetchall()

    report_type_counts = {"Upload": 0, "Live": 0}
    risk_counts = {"Alcoholic": 0, "Safe": 0, "Drunk": 0, "Sober": 0, "No Person": 0}
    driver_report_counts = {}
    for report in reports:
        dtype = report.get('detection_type') or 'Unknown'
        if dtype in report_type_counts:
            report_type_counts[dtype] += 1

        status = (report.get('status') or '').lower()
        if 'no person' in status:
            risk_counts['No Person'] += 1
        if 'alcohol:' in status:
            if 'alcoholic' in status:
                risk_counts['Alcoholic'] += 1
            elif 'safe' in status:
                risk_counts['Safe'] += 1
        if 'drunk:' in status:
            if 'drunk' in status:
                risk_counts['Drunk'] += 1
            elif 'sober' in status:
                risk_counts['Sober'] += 1

        driver_name = report.get('driver_name') or 'Unknown'
        driver_report_counts[driver_name] = driver_report_counts.get(driver_name, 0) + 1

    top_drivers = sorted(driver_report_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    top_driver_names = [item[0] for item in top_drivers]
    top_driver_counts = [item[1] for item in top_drivers]

    cursor.close()
    conn.close()

    return render_template(
        'admin-dashboard.html',
        username=current_user.username,
        users=users,
        reports=reports,
        total_users=total_users,
        total_drivers=total_drivers,
        total_chefs=total_chefs,
        total_admins=total_admins,
        report_type_counts=report_type_counts,
        risk_counts=risk_counts,
        top_driver_names=top_driver_names,
        top_driver_counts=top_driver_counts
    )

# ========================= CHEF DASHBOARD =========================
@routes.route('/chef-dashboard')
@login_required
def chef_dashboard():
    if not current_user.is_chef():
        flash("⚠️ Access denied.", "danger")
        return redirect(url_for("routes.home"))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, status, license FROM users WHERE role='driver'")
    drivers = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) AS count FROM trips WHERE status='planned'")
    active_trips = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(*) AS count FROM trips WHERE status='completed'")
    completed_trips = cursor.fetchone()['count']

    cursor.execute(
        "SELECT u.username AS driver_name, r.detection_type, r.status, r.timestamp "
        "FROM driver_detection_reports r "
        "LEFT JOIN users u ON r.driver_id=u.id "
        "ORDER BY r.timestamp DESC LIMIT 20"
    )
    reports = cursor.fetchall()

    pending_reports = len(reports)
    report_type_counts = {"Upload": 0, "Live": 0}
    trip_status_counts = {"Planned": active_trips, "Completed": completed_trips, "Other": 0}
    driver_risk_counts = {"Alcoholic": 0, "Drunk": 0, "Sober": 0, "No Person": 0}

    for report in reports:
        dtype = report.get('detection_type') or 'Unknown'
        if dtype in report_type_counts:
            report_type_counts[dtype] += 1

        status = (report.get('status') or '').lower()
        if 'alcoholic' in status:
            driver_risk_counts['Alcoholic'] += 1
        if 'drunk' in status:
            driver_risk_counts['Drunk'] += 1
        if 'sober' in status and 'drunk:' in status:
            driver_risk_counts['Sober'] += 1
        if 'no person' in status:
            driver_risk_counts['No Person'] += 1

    cursor.close()
    conn.close()

    return render_template(
        'chef-dashboard.html',
        username=current_user.username,
        drivers=drivers,
        active_trips=active_trips,
        completed_trips=completed_trips,
        pending_reports=pending_reports,
        reports=reports,
        report_type_counts=report_type_counts,
        trip_status_counts=trip_status_counts,
        driver_risk_counts=driver_risk_counts
    )

# ========================= CHEF TRIP ASSIGNMENT =========================
@routes.route('/chef/trip-assignment', methods=['GET'])
@login_required
def chef_trip_assignment():
    if not current_user.is_chef():
        flash("⚠️ Access denied.", "danger")
        return redirect(url_for("routes.home"))

    conn = get_connection()
    cursor = conn.cursor()
    
    # Get available drivers
    cursor.execute("SELECT id, username, email FROM users WHERE role='driver' ORDER BY username ASC")
    available_drivers = cursor.fetchall()
    
    # Get recent assignments by this chef
    cursor.execute(
        "SELECT t.id, t.pickup_address, t.dropoff_address, t.num_passengers, t.duration_minutes, t.status, t.assigned_at, u.username AS driver_name "
        "FROM trips t LEFT JOIN users u ON t.driver_id=u.id "
        "WHERE t.chef_id=%s ORDER BY t.assigned_at DESC LIMIT 10",
        (current_user.id,)
    )
    recent_trips = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template(
        'chef-trip-assignment.html',
        available_drivers=available_drivers,
        recent_trips=recent_trips
    )

@routes.route('/chef/assign_trip', methods=['POST'])
@login_required
def chef_assign_trip():
    if not current_user.is_chef():
        flash("⚠️ Only chefs can assign trips.", "danger")
        return redirect(url_for("routes.home"))
    
    driver_id = request.form.get('driver_id')
    pickup_address = request.form.get('pickup_address', '')
    dropoff_address = request.form.get('dropoff_address', '')
    num_passengers = request.form.get('num_passengers', 1, type=int)
    duration_minutes = request.form.get('duration_minutes', 30, type=int)
    distance_km = request.form.get('distance_km', 0, type=float)
    scheduled_time_str = request.form.get('scheduled_time', '')
    special_instructions = request.form.get('special_instructions', '')
    
    # Validation
    if not driver_id or not pickup_address or not dropoff_address or not scheduled_time_str:
        flash("⚠️ Please fill in all required fields.", "danger")
        return redirect(url_for("routes.chef_trip_assignment"))
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Verify driver exists
        cursor.execute("SELECT id FROM users WHERE id=%s AND role='driver'", (driver_id,))
        if not cursor.fetchone():
            flash("⚠️ Invalid driver selected.", "danger")
            cursor.close()
            conn.close()
            return redirect(url_for("routes.chef_trip_assignment"))
        
        # Parse scheduled time
        scheduled_time = datetime.fromisoformat(scheduled_time_str)
        
        # Create trip assignment
        cursor.execute(
            "INSERT INTO trips (user_id, chef_id, driver_id, pickup_address, dropoff_address, status, requested_at, assigned_at, scheduled_time, num_passengers, duration_minutes, distance_km, special_instructions) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (current_user.id, current_user.id, driver_id, pickup_address, dropoff_address, 'assigned', datetime.now(), datetime.now(), scheduled_time, num_passengers, duration_minutes, distance_km, special_instructions)
        )
        trip_id = cursor.lastrowid

        cursor.execute("SELECT driver_id FROM driver_locations WHERE driver_id=%s", (driver_id,))
        if cursor.fetchone():
            cursor.execute(
                "UPDATE driver_locations SET available=0, updated_at=%s WHERE driver_id=%s",
                (datetime.now(), driver_id)
            )
        else:
            cursor.execute(
                "INSERT INTO driver_locations (driver_id, latitude, longitude, available, updated_at) VALUES (%s,%s,%s,%s,%s)",
                (driver_id, 0.0, 0.0, 0, datetime.now())
            )

        conn.commit()
        
        # Get driver info for notification
        cursor.execute("SELECT id, username, email FROM users WHERE id=%s", (driver_id,))
        driver = cursor.fetchone()
        
        # Send notification to driver
        send_notification(
            driver_id, 
            'New Trip Assigned', 
            f'Chef {current_user.username} has assigned you a trip from {pickup_address} to {dropoff_address}. Scheduled for {scheduled_time.strftime("%Y-%m-%d %H:%M")}. {num_passengers} passenger(s).',
            channels=['web', 'email']
        )
        
        cursor.close()
        conn.close()
        
        flash(f"✅ Trip assigned to {driver['username']} successfully!", "success")
        return redirect(url_for("routes.chef_trip_assignment"))
        
    except Exception as e:
        flash(f"❌ Error assigning trip: {str(e)}", "danger")
        conn.close()
        return redirect(url_for("routes.chef_trip_assignment"))

# ========================= DRIVER DASHBOARD =========================
@routes.route('/driver-dashboard')
@login_required
def driver_dashboard():
    if not current_user.is_driver():
        flash("⚠️ Access denied.", "danger")
        return redirect(url_for("routes.home"))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, pickup_address AS start_location, dropoff_address AS end_location, requested_at AS start_time, completed_at AS end_time, status, distance_km, duration_minutes, driver_id "
        "FROM trips WHERE driver_id=%s ORDER BY requested_at DESC",
        (current_user.id,)
    )
    trips = cursor.fetchall()

    total_trips = len(trips)
    total_distance = sum(t.get("distance_km",0) for t in trips)
    total_minutes = sum(t.get("duration_minutes",0) for t in trips)
    driving_hours = f"{total_minutes//60}h {total_minutes%60}m"

    trip_status_counts = {}
    trip_distance_labels = []
    trip_distance_values = []
    for index, trip in enumerate(trips[-10:], start=1):
        status = trip.get('status') or 'Unknown'
        trip_status_counts[status] = trip_status_counts.get(status, 0) + 1
        label = f"{trip.get('start_location','Trip')}"
        trip_distance_labels.append(label)
        trip_distance_values.append(trip.get('distance_km', 0))

    cursor.execute(
        "SELECT license, fuel_type, length, service_date FROM vehicles WHERE driver_id=%s LIMIT 1",
        (current_user.id,)
    )
    vehicle = cursor.fetchone()
    if not vehicle:
        vehicle = {"license":"N/A","fuel_type":"N/A","length":"N/A","service_date":"N/A"}

    cursor.execute(
        "SELECT detection_type, status, timestamp FROM driver_detection_reports "
        "WHERE driver_id=%s ORDER BY timestamp DESC LIMIT 10",
        (current_user.id,)
    )
    reports = cursor.fetchall()

    detection_status_counts = {"No Person": 0, "Alcohol Detected": 0, "Safe / Non-Alcoholic": 0}
    recent_detection_labels = []
    recent_detection_values = []
    for report in reports:
        status_text = report.get('status') or ''
        if 'no person' in status_text.lower():
            detection_status_counts['No Person'] += 1
        elif 'alcoholic' in status_text.lower() or 'drunk' in status_text.lower():
            detection_status_counts['Alcohol Detected'] += 1
        else:
            detection_status_counts['Safe / Non-Alcoholic'] += 1

        timestamp = report.get('timestamp')
        if isinstance(timestamp, datetime):
            recent_detection_labels.append(timestamp.strftime('%d %b %H:%M'))
        else:
            recent_detection_labels.append(str(timestamp))
        recent_detection_values.append(1)

    current_trip = next((t for t in trips if t['status'] in ('assigned','accepted','ongoing')), None)
    cursor.close()
    conn.close()

    return render_template(
        "driver-dashboard.html",
        username=current_user.username,
        trips=trips,
        total_trips=total_trips,
        total_distance=total_distance,
        driving_hours=driving_hours,
        vehicle=vehicle,
        current_trip=current_trip,
        reports=reports,
        detection_status_counts=detection_status_counts,
        recent_detection_labels=recent_detection_labels,
        recent_detection_values=recent_detection_values,
        trip_status_counts=trip_status_counts,
        trip_distance_labels=trip_distance_labels,
        trip_distance_values=trip_distance_values
    )

# ========================= DRIVER ASSIGNED TRIPS JSON =========================
@routes.route('/driver/assigned_trips_json')
@login_required
def driver_assigned_trips_json():
    if not current_user.is_driver():
        return jsonify([])

    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Select all trip details
        query = (
            "SELECT t.id, t.pickup_address, t.dropoff_address, "
            "t.scheduled_time, t.num_passengers, t.duration_minutes, t.distance_km, "
            "t.special_instructions, t.status, t.assigned_at, "
            "u.username AS chef_name, u.email AS chef_email "
            "FROM trips t LEFT JOIN users u ON t.chef_id=u.id "
            "WHERE t.driver_id=%s AND t.status IN ('assigned','accepted','ongoing','completed','rejected','cancelled') "
            "ORDER BY t.assigned_at DESC"
        )
        
        cursor.execute(query, (current_user.id,))
        trips = cursor.fetchall()
        cursor.close()
        conn.close()

        # Format timestamps
        for trip in trips:
            if isinstance(trip.get('assigned_at'), datetime):
                trip['assigned_at'] = trip['assigned_at'].strftime('%Y-%m-%d %H:%M')
            if isinstance(trip.get('scheduled_time'), datetime):
                trip['scheduled_time'] = trip['scheduled_time'].strftime('%Y-%m-%d %H:%M')
            
        return jsonify(trips)
    
    except Exception as e:
        print(f"[ERROR] driver_assigned_trips_json: {e}")
        return jsonify({'error': str(e)}), 500

# ========================= MESSAGING =========================
@routes.route('/messages')
@login_required
def messages():
    conn = get_connection()
    cursor = conn.cursor()

    drivers = []
    if current_user.is_chef() or current_user.is_admin():
        cursor.execute("SELECT id, username, email FROM users WHERE role='driver' ORDER BY email ASC")
        drivers = cursor.fetchall()

    columns = get_message_column_names()
    cursor.execute(
        f"SELECT COUNT(*) AS unread FROM messages WHERE receiver_id=%s AND {columns['status']}='unread'",
        (current_user.id,)
    )
    unread_count = cursor.fetchone().get('unread', 0)

    cursor.close()
    conn.close()

    return render_template(
        'messages.html',
        username=current_user.username,
        role=current_user.role,
        user_id=current_user.id,
        drivers=drivers,
        unread_count=unread_count
    )

@routes.route('/messages/send', methods=['POST'])
@login_required
def send_message():
    receiver_email = request.form.get('receiver_email', '').strip().lower()
    subject = request.form.get('subject', '').strip()
    body = request.form.get('body', '').strip()

    if not body or not receiver_email:
        flash('⚠️ Please provide both recipient email and a message body.', 'danger')
        return redirect(url_for('routes.messages'))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users WHERE email=%s", (receiver_email,))
    receiver = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not receiver:
        flash('❌ Driver email not found.', 'danger')
        return redirect(url_for('routes.messages'))

    if current_user.is_chef() or current_user.is_admin():
        if receiver['role'] != 'driver':
            flash('❌ Messages may only be sent to drivers from this form.', 'danger')
            return redirect(url_for('routes.messages'))

    # Redirect to Gmail compose with pre-filled fields
    gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={quote(receiver_email)}&su={quote(subject or 'SafeDrive Message')}&body={quote(body)}"
    return redirect(gmail_url)

@routes.route('/messages/inbox_json')
@login_required
def messages_inbox_json():
    conn = get_connection()
    cursor = conn.cursor()
    columns = get_message_column_names()
    columns = get_message_column_names()
    subject_select = f"m.{columns['subject']} AS subject, " if columns.get('subject') else "'' AS subject, "
    cursor.execute(
        f"SELECT m.id, {subject_select} m.{columns['body']} AS body, m.{columns['status']} AS status, m.{columns['created_at']} AS created_at, u.username AS sender_name, u.email AS sender_email, u.role AS sender_role "
        f"FROM messages m JOIN users u ON m.sender_id=u.id "
        f"WHERE m.receiver_id=%s ORDER BY m.{columns['created_at']} DESC LIMIT 100",
        (current_user.id,)
    )
    inbox = cursor.fetchall()
    cursor.close()
    conn.close()

    for msg in inbox:
        if isinstance(msg.get('created_at'), datetime):
            msg['created_at'] = msg['created_at'].strftime('%Y-%m-%d %H:%M')
        msg['preview'] = (msg['body'][:120] + '...') if len(msg['body']) > 120 else msg['body']
    return jsonify(inbox)

@routes.route('/messages/sent_json')
@login_required
def messages_sent_json():
    conn = get_connection()
    cursor = conn.cursor()
    columns = get_message_column_names()
    columns = get_message_column_names()
    subject_select = f"m.{columns['subject']} AS subject, " if columns.get('subject') else "'' AS subject, "
    cursor.execute(
        f"SELECT m.id, {subject_select} m.{columns['body']} AS body, m.{columns['status']} AS status, m.{columns['created_at']} AS created_at, u.username AS receiver_name, u.email AS receiver_email, u.role AS receiver_role "
        f"FROM messages m JOIN users u ON m.receiver_id=u.id "
        f"WHERE m.sender_id=%s ORDER BY m.{columns['created_at']} DESC LIMIT 100",
        (current_user.id,)
    )
    sent = cursor.fetchall()
    cursor.close()
    conn.close()

    for msg in sent:
        if isinstance(msg.get('created_at'), datetime):
            msg['created_at'] = msg['created_at'].strftime('%Y-%m-%d %H:%M')
        msg['preview'] = (msg['body'][:120] + '...') if len(msg['body']) > 120 else msg['body']
    return jsonify(sent)

@routes.route('/messages/view/<int:message_id>')
@login_required
def messages_view(message_id):
    conn = get_connection()
    cursor = conn.cursor()
    columns = get_message_column_names()
    columns = get_message_column_names()
    subject_select = f"m.{columns['subject']} AS subject, " if columns.get('subject') else "'' AS subject, "
    cursor.execute(
        f"SELECT m.id, {subject_select} m.{columns['body']} AS body, m.{columns['status']} AS status, m.{columns['created_at']} AS created_at, m.sender_id, m.receiver_id, u.username AS sender_name, u.email AS sender_email, u.role AS sender_role, "
        f"v.username AS receiver_name, v.email AS receiver_email, v.role AS receiver_role "
        f"FROM messages m "
        f"JOIN users u ON m.sender_id=u.id "
        f"JOIN users v ON m.receiver_id=v.id "
        f"WHERE m.id=%s",
        (message_id,)
    )
    msg = cursor.fetchone()
    if not msg or current_user.id not in (msg['sender_id'], msg['receiver_id']):
        cursor.close()
        conn.close()
        return jsonify({'error': 'Message not found or access denied.'}), 404

    if msg['receiver_id'] == current_user.id and msg['status'] == 'unread':
        cursor.execute(f"UPDATE messages SET {columns['status']}='read' WHERE id=%s", (message_id,))
        conn.commit()
        msg['status'] = 'read'

    cursor.close()
    conn.close()

    if isinstance(msg.get('created_at'), datetime):
        msg['created_at'] = msg['created_at'].strftime('%Y-%m-%d %H:%M')
    return jsonify(msg)

@routes.route('/messages/unread_count')
@login_required
def messages_unread_count():
    conn = get_connection()
    cursor = conn.cursor()
    columns = get_message_column_names()
    cursor.execute(
        f"SELECT COUNT(*) AS count FROM messages WHERE receiver_id=%s AND {columns['status']}='unread'",
        (current_user.id,)
    )
    count = cursor.fetchone().get('count', 0)
    cursor.close()
    conn.close()
    return jsonify({'count': count})

# ========================= PASSENGER DASHBOARD =========================
@routes.route('/passenger-dashboard')
@login_required
def passenger_dashboard():
    if not current_user.is_passenger():
        flash('⚠️ Access denied.', 'danger')
        return redirect(url_for('routes.home'))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT t.*, u.username AS driver_name, u.email AS driver_email "
        "FROM trips t LEFT JOIN users u ON t.driver_id=u.id "
        "WHERE t.user_id=%s ORDER BY t.requested_at DESC LIMIT 10",
        (current_user.id,)
    )
    trips = cursor.fetchall()

    active_trip = next((t for t in trips if t['status'] in ('pending','assigned','accepted','ongoing')), None)
    cursor.execute("SELECT COUNT(*) AS count FROM driver_locations WHERE available=1")
    available_drivers = cursor.fetchone()['count']

    cursor.close()
    conn.close()
    return render_template(
        'passenger-dashboard.html',
        username=current_user.username,
        trips=trips,
        active_trip=active_trip,
        available_drivers=available_drivers
    )

@routes.route('/passenger/request_ride', methods=['POST'])
@login_required
def request_ride():
    if not current_user.is_passenger():
        return jsonify({'success': False, 'message': 'Only passengers can request rides.'}), 403

    data = request.get_json() or {}
    pickup_address = data.get('pickup_address', 'Pickup Point')
    dropoff_address = data.get('dropoff_address', 'Dropoff Point')
    pickup_lat = float(data.get('pickup_lat', 12.9716))
    pickup_lng = float(data.get('pickup_lng', 77.5946))
    dropoff_lat = float(data.get('dropoff_lat', 12.9352))
    dropoff_lng = float(data.get('dropoff_lng', 77.6245))

    trip_id, driver_info = assign_driver_to_trip(
        current_user.id,
        pickup_address,
        dropoff_address,
        pickup_lat,
        pickup_lng,
        dropoff_lat,
        dropoff_lng
    )

    if driver_info:
        send_notification(current_user.id, 'Ride Confirmed', f'Your ride request has been assigned to {driver_info["username"]}.', channels=['web', 'email'])
        return jsonify({'success': True, 'trip_id': trip_id, 'assigned': True, 'driver': {'id': driver_info['driver_id'], 'name': driver_info['username']}})

    send_notification(current_user.id, 'Ride Pending', 'No drivers are available right now; your request is queued.', channels=['web', 'email'])
    return jsonify({'success': True, 'trip_id': trip_id, 'assigned': False})

@routes.route('/trip/track/<int:trip_id>')
@login_required
def track_trip(trip_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trips WHERE id=%s", (trip_id,))
    trip = cursor.fetchone()

    if not trip or (trip['user_id'] != current_user.id and trip.get('driver_id') != current_user.id and not current_user.is_admin()):
        flash('⚠️ Trip not found or access denied.', 'danger')
        return redirect(url_for('routes.home'))

    cursor.execute("SELECT username FROM users WHERE id=%s", (trip.get('driver_id'),))
    driver = cursor.fetchone()
    cursor.close()
    return render_template('trip-tracking.html', trip=trip, driver=driver)

@routes.route('/trip/track_data/<int:trip_id>')
@login_required
def trip_track_data(trip_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trips WHERE id=%s", (trip_id,))
    trip = cursor.fetchone()
    if not trip or (trip['user_id'] != current_user.id and trip.get('driver_id') != current_user.id and not current_user.is_admin()):
        cursor.close()
        conn.close()
        return jsonify({}), 404

    driver_location = None
    if trip.get('driver_id'):
        cursor.execute("SELECT latitude, longitude FROM driver_locations WHERE driver_id=%s", (trip['driver_id'],))
        driver_location = cursor.fetchone()

    current_eta = trip.get('eta_minutes') or 0
    current_status = trip.get('status')

    cursor.close()
    conn.close()
    return jsonify({
        'trip': trip,
        'driver_location': driver_location,
        'status': current_status,
        'eta_minutes': current_eta
    })

@routes.route('/trip/update_status', methods=['POST'])
@login_required
def update_trip_status():
    data = request.get_json() or {}
    trip_id = data.get('trip_id')
    target_status = data.get('status')

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trips WHERE id=%s", (trip_id,))
    trip = cursor.fetchone()
    if not trip:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'message': 'Trip not found.'}), 404

    actor_is_driver = current_user.is_driver() and trip.get('driver_id') == current_user.id
    actor_is_passenger = current_user.is_passenger() and trip.get('user_id') == current_user.id

    allowed_transitions = {
        'pending': ['cancelled'],
        'assigned': ['accepted', 'rejected', 'cancelled'],
        'accepted': ['ongoing', 'cancelled'],
        'ongoing': ['completed', 'cancelled']
    }

    current_status = trip['status']
    if target_status not in allowed_transitions.get(current_status, []):
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'message': f'Cannot transition from {current_status} to {target_status}.'}), 400

    if target_status in ['accepted', 'ongoing', 'completed'] and not actor_is_driver:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'message': 'Only the assigned driver may update this status.'}), 403
    if target_status == 'cancelled' and not (actor_is_driver or actor_is_passenger or current_user.is_admin()):
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'message': 'Not authorized to cancel this trip.'}), 403

    time_column = None
    if target_status == 'accepted':
        time_column = 'accepted_at'
    elif target_status == 'ongoing':
        time_column = 'started_at'
    elif target_status == 'completed':
        time_column = 'completed_at'
    elif target_status == 'rejected':
        time_column = 'rejected_at'

    if time_column:
        cursor.execute(f"UPDATE trips SET status=%s, {time_column}=%s WHERE id=%s", (target_status, datetime.now(), trip_id))
    else:
        cursor.execute("UPDATE trips SET status=%s WHERE id=%s", (target_status, trip_id))

    if target_status in ['completed', 'cancelled', 'rejected'] and trip.get('driver_id'):
        cursor.execute("UPDATE driver_locations SET available=1, updated_at=%s WHERE driver_id=%s", (datetime.now(), trip['driver_id']))

    conn.commit()
    cursor.close()
    conn.close()

    passenger_message = f'Trip {target_status} by driver.' if actor_is_driver else f'Trip {target_status}.'
    driver_message = f'Driver has {target_status} the trip.'
    if target_status == 'completed':
        driver_message = 'Trip completed successfully.'
    if target_status == 'cancelled':
        driver_message = 'Trip was cancelled.'

    if trip.get('driver_id'):
        send_notification(trip['driver_id'], 'Trip Update', driver_message, channels=['web', 'email'])
    if trip.get('chef_id') and trip.get('chef_id') != trip.get('driver_id'):
        send_notification(trip['chef_id'], 'Trip Update', f'Trip {trip_id} was {target_status}.', channels=['web', 'email'])
    send_notification(trip['user_id'], 'Trip Update', passenger_message, channels=['web', 'email'])

    return jsonify({'success': True, 'status': target_status})

# ========================= DRIVER LOCATION =========================
@routes.route('/driver/update_location', methods=['POST'])
@login_required
def driver_update_location():
    if not current_user.is_driver():
        return jsonify({'success': False, 'message': 'Only drivers can update location.'}), 403

    data = request.get_json() or {}
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    available = data.get('available', True)

    if latitude is None or longitude is None:
        return jsonify({'success': False, 'message': 'Missing GPS coordinates.'}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT driver_id FROM driver_locations WHERE driver_id=%s", (current_user.id,))
    if cursor.fetchone():
        cursor.execute(
            "UPDATE driver_locations SET latitude=%s, longitude=%s, available=%s, updated_at=%s WHERE driver_id=%s",
            (latitude, longitude, bool(available), datetime.now(), current_user.id)
        )
    else:
        cursor.execute(
            "INSERT INTO driver_locations (driver_id, latitude, longitude, available, updated_at) VALUES (%s,%s,%s,%s,%s)",
            (current_user.id, latitude, longitude, bool(available), datetime.now())
        )
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'success': True})

# ========================= NOTIFICATIONS =========================
@routes.route('/notifications/poll')
@login_required
def notifications_poll():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, channel, title, message, read_status, created_at FROM notifications "
        "WHERE user_id=%s ORDER BY created_at DESC LIMIT 20",
        (current_user.id,)
    )
    notes = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(notes)

# ========================= DRIVER PROFILE & VERIFICATION =========================
@routes.route('/driver/profile', methods=['GET','POST'])
@login_required
def driver_profile():
    if not current_user.is_driver():
        flash('⚠️ Access denied.', 'danger')
        return redirect(url_for('routes.home'))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM driver_profiles WHERE user_id=%s", (current_user.id,))
    profile = cursor.fetchone()

    if request.method == 'POST':
        phone = request.form.get('phone')
        license_number = request.form.get('license_number')
        file = request.files.get('license_file')
        filename = None

        if file:
            upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'uploads', 'licenses')
            os.makedirs(upload_folder, exist_ok=True)
            filename = secure_filename(file.filename)
            file.save(os.path.join(upload_folder, filename))

        if profile:
            cursor.execute(
                "UPDATE driver_profiles SET phone=%s, license_number=%s, license_file=COALESCE(%s, license_file), verification_status='pending', updated_at=%s WHERE user_id=%s",
                (phone, license_number, filename, datetime.now(), current_user.id)
            )
        else:
            cursor.execute(
                "INSERT INTO driver_profiles (user_id, phone, license_number, license_file, verification_status, trust_level, updated_at) VALUES (%s,%s,%s,%s,'pending',0,%s)",
                (current_user.id, phone, license_number, filename, datetime.now())
            )
        conn.commit()
        cursor.close()
        conn.close()
        flash('✅ Profile updated and verification submitted.', 'success')
        return redirect(url_for('routes.driver_profile'))

    cursor.close()
    conn.close()
    return render_template('driver-profile.html', profile=profile)

@routes.route('/admin/driver-verifications')
@login_required
def admin_driver_verifications():
    if not current_user.is_admin():
        flash('⚠️ Access denied.', 'danger')
        return redirect(url_for('routes.home'))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT p.id, p.user_id, p.phone, p.license_number, p.license_file, p.verification_status, p.trust_level, p.compliance_notes, u.username, u.email "
        "FROM driver_profiles p JOIN users u ON p.user_id=u.id"
    )
    profiles = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('admin-driver-verifications.html', profiles=profiles)

@routes.route('/admin/driver-verification/<int:profile_id>/<action>')
@login_required
def admin_update_verification(profile_id, action):
    if not current_user.is_admin():
        flash('⚠️ Access denied.', 'danger')
        return redirect(url_for('routes.home'))

    if action not in ('approve', 'reject'):
        flash('❌ Invalid action.', 'danger')
        return redirect(url_for('routes.admin_driver_verifications'))

    new_status = 'approved' if action == 'approve' else 'rejected'
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE driver_profiles SET verification_status=%s, updated_at=%s WHERE id=%s", (new_status, datetime.now(), profile_id))
    cursor.execute("SELECT user_id FROM driver_profiles WHERE id=%s", (profile_id,))
    row = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()

    if row:
        send_notification(row['user_id'], 'Verification ' + new_status.title(), f'Your driver profile has been {new_status}.', channels=['web', 'email'])

    flash(f'✅ Driver verification {new_status}.', 'success')
    return redirect(url_for('routes.admin_driver_verifications'))

# ========================= HELPER: SAVE DETECTION REPORT =========================
def save_detection_report(driver_id, detection_type, status):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO driver_detection_reports (driver_id,detection_type,status,timestamp) "
        "VALUES (%s,%s,%s,%s)",
        (driver_id,detection_type,status,datetime.now())
    )
    conn.commit()
    cursor.close()
    conn.close()

# ========================= UPLOAD IMAGE DETECTION =========================
@routes.route("/upload_image", methods=["POST"])
@login_required
def upload_image():
    if not current_user.is_driver():
        return jsonify({"status":"error","message":"Only drivers can detect."})

    load_models()  # lazy load models
    file = request.files.get("file")
    if not file:
        return jsonify({"status":"error","message":"No file uploaded."})

    img = Image.open(file.stream).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    persons = detect_person(img_cv)
    if not persons:
        save_detection_report(current_user.id,"Upload","No Person")
        return jsonify({"status":"no_person"})

    cropped, bbox = persons[0]

    processed_driver = preprocess_image(cropped)
    pred_driver = driver_model.predict(processed_driver)[0]
    driver_status = "alcoholic" if np.argmax(pred_driver)==0 else "safe"

    processed_drunk = preprocess_image(cropped)
    pred_drunk = drunk_model.predict(processed_drunk)[0]
    drunk_status = "drunk" if np.argmax(pred_drunk)==0 else "sober"

    save_detection_report(current_user.id,"Upload",f"Alcohol:{driver_status},Drunk:{drunk_status}")
    return jsonify({"driver_status":driver_status,"drunk_status":drunk_status})

# ========================= LIVE CAMERA DETECTION =========================
@routes.route("/live_detect", methods=["POST"])
@login_required
def live_detect():
    if not current_user.is_driver():
        return jsonify({"status":"error","message":"Only drivers can detect."})

    load_models()  # lazy load models
    file = request.files.get("frame")
    if not file:
        return jsonify({"status":"error","message":"No frame uploaded."})

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    persons = detect_person(frame)
    if not persons:
        save_detection_report(current_user.id,"Live","No Person")
        return jsonify({"status":"align_face"})

    cropped, bbox = persons[0]

    processed_driver = preprocess_image(cropped)
    pred_driver = driver_model.predict(processed_driver)[0]
    driver_status = "alcoholic" if np.argmax(pred_driver)==0 else "safe"

    processed_drunk = preprocess_image(cropped)
    pred_drunk = drunk_model.predict(processed_drunk)[0]
    drunk_status = "drunk" if np.argmax(pred_drunk)==0 else "sober"

    save_detection_report(current_user.id,"Live",f"Alcohol:{driver_status},Drunk:{drunk_status}")
    return jsonify({"driver_status":driver_status,"drunk_status":drunk_status})

# ========================= DRIVER REPORTS JSON =========================
@routes.route('/driver/reports_json')
@login_required
def driver_reports_json():
    if not current_user.is_driver():
        return jsonify([])

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp,detection_type,status FROM driver_detection_reports "
        "WHERE driver_id=%s ORDER BY timestamp DESC LIMIT 10",
        (current_user.id,)
    )
    reports = cursor.fetchall()
    cursor.close()
    conn.close()

    for r in reports:
        if isinstance(r['timestamp'], datetime):
            r['timestamp'] = r['timestamp'].strftime('%d/%m/%Y %H:%M:%S')

    return jsonify(reports)


# ========================= UPDATE USER ROLE =========================
@routes.route('/update_role', methods=['POST'])
@login_required
def update_role():
    if not current_user.is_admin():
        flash("⚠️ Access denied.", "danger")
        return redirect(url_for("routes.home"))

    user_id = request.form.get('user_id')
    new_role = request.form.get('role')

    if not user_id or new_role not in ['driver', 'chef', 'admin']:
        flash("❌ Invalid data.", "danger")
        return redirect(url_for('routes.admin_dashboard'))

    conn = get_connection()
    cursor = conn.cursor()
    if int(user_id) == int(current_user.id):
        flash("⚠️ You cannot change your own role.", "warning")
        cursor.close()
        conn.close()
        return redirect(url_for('routes.admin_dashboard'))

    cursor.execute("UPDATE users SET role=%s WHERE id=%s", (new_role, user_id))
    conn.commit()
    cursor.close()
    conn.close()

    flash(f"✅ User role updated to {new_role}.", "success")
    return redirect(url_for('routes.admin_dashboard'))

# ========================= DELETE USER =========================
@routes.route('/delete_user', methods=['POST'])
@login_required
def delete_user():
    if not current_user.is_admin():
        flash("⚠️ Access denied.", "danger")
        return redirect(url_for("routes.home"))

    user_id = request.form.get('user_id')
    if not user_id or int(user_id) == int(current_user.id):
        flash("⚠️ Invalid request or cannot delete yourself.", "danger")
        return redirect(url_for('routes.admin_dashboard'))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
    conn.commit()
    cursor.close()
    conn.close()

    flash("✅ User removed successfully.", "success")
    return redirect(url_for('routes.admin_dashboard'))


# ========================= EMAIL COMPOSITION - Gmail Integration =========================
@routes.route('/email/drivers_json')
@login_required
def email_drivers_json():
    """Fetch all drivers for email recipient selection - Chef/Admin only"""
    if not (current_user.is_chef() or current_user.is_admin()):
        return jsonify({'error': 'Access denied'}), 403

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email FROM users WHERE role='driver' ORDER BY username ASC"
        )
        drivers = cursor.fetchall()
    except Exception as exc:
        if current_app:
            current_app.logger.exception('Failed to load drivers for email compose')
        return jsonify({'error': 'Unable to load drivers. Please try again later.'}), 500
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    drivers_list = [
        {'id': d['id'], 'username': d['username'], 'email': d['email']}
        for d in drivers
    ]
    return jsonify(drivers_list)


@routes.route('/email/compose')
@login_required
def email_compose():
    """Display email composition form - Chef/Admin only"""
    if not (current_user.is_chef() or current_user.is_admin()):
        flash('⚠️ Access denied.', 'danger')
        return redirect(url_for('routes.home'))
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, email FROM users WHERE role='driver' ORDER BY username ASC"
    )
    drivers = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('email_compose.html', drivers=drivers, current_user=current_user)