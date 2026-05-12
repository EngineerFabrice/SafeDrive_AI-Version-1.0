import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='safedrive_ai', port=3307)
cursor = conn.cursor()

# Add missing columns if they don't exist
try:
    cursor.execute("ALTER TABLE trips ADD COLUMN chef_id INT NULL AFTER user_id")
    print("✅ Added chef_id column")
except:
    print("⚠️ chef_id column already exists")

try:
    cursor.execute("ALTER TABLE trips ADD COLUMN num_passengers INT DEFAULT 1 AFTER duration_minutes")
    print("✅ Added num_passengers column")
except:
    print("⚠️ num_passengers column already exists")

try:
    cursor.execute("ALTER TABLE trips ADD COLUMN scheduled_time DATETIME NULL AFTER assigned_at")
    print("✅ Added scheduled_time column")
except:
    print("⚠️ scheduled_time column already exists")

try:
    cursor.execute("ALTER TABLE trips ADD COLUMN special_instructions TEXT NULL AFTER scheduled_time")
    print("✅ Added special_instructions column")
except:
    print("⚠️ special_instructions column already exists")

try:
    cursor.execute("ALTER TABLE trips ADD COLUMN accepted_at TIMESTAMP NULL AFTER assigned_at")
    print("✅ Added accepted_at column")
except:
    print("⚠️ accepted_at column already exists")

try:
    cursor.execute("ALTER TABLE trips ADD COLUMN started_at TIMESTAMP NULL AFTER accepted_at")
    print("✅ Added started_at column")
except:
    print("⚠️ started_at column already exists")

try:
    cursor.execute("ALTER TABLE trips ADD COLUMN rejected_at TIMESTAMP NULL AFTER started_at")
    print("✅ Added rejected_at column")
except:
    print("⚠️ rejected_at column already exists")

# Add foreign key for chef_id
try:
    cursor.execute("ALTER TABLE trips ADD CONSTRAINT fk_trips_chef_id FOREIGN KEY (chef_id) REFERENCES users(id)")
    print("✅ Added chef_id foreign key")
except:
    print("⚠️ chef_id foreign key already exists")

try:
    cursor.execute(
        "ALTER TABLE trips MODIFY COLUMN status ENUM('requested', 'assigned', 'accepted', 'ongoing', 'completed', 'cancelled', 'rejected') DEFAULT 'requested'"
    )
    print("✅ Updated status enum values")
except Exception as e:
    print(f"⚠️ status enum update skipped: {e}")

try:
    cursor.execute("ALTER TABLE notifications ADD COLUMN channel ENUM('email','sms','push','web') DEFAULT 'web' AFTER user_id")
    print("✅ Added notifications.channel column")
except Exception as e:
    print(f"⚠️ notifications.channel column already exists or could not be added: {e}")
try:
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INT AUTO_INCREMENT PRIMARY KEY, "
        "sender_id INT NOT NULL, "
        "receiver_id INT NOT NULL, "
        "subject VARCHAR(255) NULL, "
        "body TEXT NOT NULL, "
        "status ENUM('unread','read') DEFAULT 'unread', "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "FOREIGN KEY (sender_id) REFERENCES users(id), "
        "FOREIGN KEY (receiver_id) REFERENCES users(id))"
    )
    print("✅ Created messages table")
except Exception as e:
    print(f"⚠️ messages table already exists or could not be created: {e}")

try:
    cursor.execute("ALTER TABLE messages ADD COLUMN subject VARCHAR(255) NULL AFTER receiver_id")
    print("✅ Added messages.subject column")
except Exception as e:
    print(f"⚠️ messages.subject column already exists or could not be added: {e}")

# Migrate legacy message schema if needed
try:
    cursor.execute("SHOW COLUMNS FROM messages LIKE 'content'")
    if cursor.fetchone():
        cursor.execute("ALTER TABLE messages CHANGE COLUMN content body TEXT NOT NULL")
        print("✅ Renamed messages.content to messages.body")
except Exception as e:
    print(f"⚠️ Could not rename messages.content: {e}")

try:
    cursor.execute("SHOW COLUMNS FROM messages LIKE 'sent_at'")
    if cursor.fetchone():
        cursor.execute("ALTER TABLE messages CHANGE COLUMN sent_at created_at DATETIME NULL DEFAULT CURRENT_TIMESTAMP")
        print("✅ Renamed messages.sent_at to messages.created_at")
except Exception as e:
    print(f"⚠️ Could not rename messages.sent_at: {e}")

try:
    cursor.execute("SHOW COLUMNS FROM messages LIKE 'read_status'")
    if cursor.fetchone():
        cursor.execute("ALTER TABLE messages CHANGE COLUMN read_status status ENUM('unread','read') DEFAULT 'unread'")
        print("✅ Renamed messages.read_status to messages.status")
except Exception as e:
    print(f"⚠️ Could not rename messages.read_status: {e}")
conn.commit()
cursor.close()
conn.close()

print("\n✅ Schema update completed!")
