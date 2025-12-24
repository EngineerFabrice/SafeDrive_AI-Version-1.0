# website/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from . import User, get_connection, bcrypt
from .yolo_detector import detect_person
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import threading

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
            (username,email,hashed_password,'driver')
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

    # Fetch recent driver detection reports
    cursor.execute(
        "SELECT driver_id, detection_type, status, timestamp "
        "FROM driver_detection_reports ORDER BY timestamp DESC LIMIT 5"
    )
    reports = cursor.fetchall()
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
        total_admins=total_admins
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
        "SELECT driver_id, detection_type, status, timestamp "
        "FROM driver_detection_reports ORDER BY timestamp DESC LIMIT 5"
    )
    reports = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template(
        'chef-dashboard.html',
        username=current_user.username,
        drivers=drivers,
        active_trips=active_trips,
        completed_trips=completed_trips,
        reports=reports
    )

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
        "SELECT start_location,end_location,start_time,end_time,status,distance_km,duration_minutes "
        "FROM trips WHERE user_id=%s ORDER BY start_time DESC",
        (current_user.id,)
    )
    trips = cursor.fetchall()

    total_trips = len(trips)
    total_distance = sum(t.get("distance_km",0) for t in trips)
    total_minutes = sum(t.get("duration_minutes",0) for t in trips)
    driving_hours = f"{total_minutes//60}h {total_minutes%60}m"

    cursor.execute(
        "SELECT license, fuel_type, length, service_date FROM vehicles WHERE driver_id=%s LIMIT 1",
        (current_user.id,)
    )
    vehicle = cursor.fetchone()
    if not vehicle:
        vehicle = {"license":"N/A","fuel_type":"N/A","length":"N/A","service_date":"N/A"}

    cursor.execute(
        "SELECT detection_type, status, timestamp FROM driver_detection_reports "
        "WHERE driver_id=%s ORDER BY timestamp DESC LIMIT 5",
        (current_user.id,)
    )
    reports = cursor.fetchall()

    current_trip = next((t for t in trips if t['status'].lower()=="en route"), None)
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
        reports=reports
    )

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