from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import hashlib
from datetime import datetime




app = Flask(__name__)
CORS(app)

# Database configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "deepesh3#",  # Replace with your MySQL password
    "database": "crowd_management_db"
}



def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Validate role-based unique key decisions
        if data['role'] == 'admin':
            primary_key_field = 'dept_id'
            primary_key_value = data.get('deptId')
            if not primary_key_value:
                return jsonify({"success": False, "message": "Department ID is required for admin."}), 400
        elif data['role'] == 'user':
            primary_key_field = 'aadhar'
            primary_key_value = data.get('aadhar')
            if not primary_key_value:
                return jsonify({"success": False, "message": "Aadhar Card number is required for user."}), 400
        else:
            return jsonify({"success": False, "message": "Invalid role provided."}), 400

        sql = """
            INSERT INTO users (
                role, first_name, middle_name, last_name, dob, gender, dept_name, dept_id, aadhar, address, state, country,
                email, password, mobile, alt_mobile, security_question, security_answer
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            data['role'],
            data.get('firstName'),
            data.get('middleName'),
            data.get('lastName'),
            data.get('dob'),
            data.get('gender'),
            data.get('deptName') if data['role'] == 'admin' else None,
            data.get('deptId') if data['role'] == 'admin' else None,
            data.get('aadhar') if data['role'] == 'user' else None,
            data.get('address'),
            data.get('state'),
            data.get('country'),
            data.get('email'),
            hash_password(data.get('password')),
            data.get('mobile'),
            data.get('altMobile'),
            data.get('securityQuestion'),
            data.get('securityAnswer')
        )

        cursor.execute(sql, values)
        conn.commit()

        return jsonify({"success": True, "message": "Signup successful."})

    # except mysql.connector.IntegrityError as ie:
    #     # Likely due to unique constraint violation on primary key or email
    #     if 'dept_id' in str(ie) or 'aadhar' in str(ie):
    #         return jsonify({"success": False, "message": f"{primary_key_field.replace('_', ' ').title()} already exists."}), 400
    #     elif 'email' in str(ie):
    #         return jsonify({"success": False, "message": "Email already exists."}), 400
    #     else:
    #         print(ie)
    #         return jsonify({"success": False, "message": "Integrity error during signup."}), 400
    # except Exception as e:
    #     print(e)
    #     return jsonify({"success": False, "message": "Signup failed due to server error."}), 500

    except mysql.connector.IntegrityError as ie:
        error_message = str(ie)
        print("IntegrityError caught:", error_message)  # For server logs

        # Default message
        user_message = "Integrity error during signup."

        # Detect which field triggered the duplicate error
        if "email" in error_message or "PRIMARY" in error_message:
            # In your table, PRIMARY KEY = email
            user_message = "Email already exists."
        elif "aadhar" in error_message:
            user_message = "Aadhar number already exists."
        elif "dept_id" in error_message:
            user_message = "Department ID already exists."
        elif "mobile" in error_message:
            user_message = "Mobile number already exists."
        elif "alt_mobile" in error_message:
            user_message = "Alternate mobile number already exists."
        elif "Duplicate entry" in error_message:
            user_message = "Duplicate entry detected in the database."

        return jsonify({"success": False, "message": user_message}), 400

    
    except Exception as e:
        print("General Exception:", e)  # <-- This shows in terminal
        return jsonify({"success": False, "message": "Signup failed due to server error."}), 500
    

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

 

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    role = data.get('role')
    username = data.get('username')  # For admin, username is dept_id; for user, it's email
    password = data.get('password')
    
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        if role == 'admin':
            # Query for admin authentication using dept_id
            sql = """
                SELECT * FROM users 
                WHERE dept_id = %s AND password = %s AND role = 'admin'
            """
            cursor.execute(sql, (username, hash_password(password)))
        else:
            # Query for regular user authentication using email
            sql = """
                SELECT * FROM users 
                WHERE email = %s AND password = %s AND role = 'user'
            """
            cursor.execute(sql, (username, hash_password(password)))
        
        user = cursor.fetchone()
        
        if user:
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': {
                    'role': user['role'],
                    'name': f"{user['first_name']} {user['last_name']}",
                    'dept_id': user.get('dept_id'),
                    'aadhar': user.get('aadhar'),
                    'email': user['email']
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid username or password. Please try again.'
            }), 401
            
    except mysql.connector.Error as err:
        return jsonify({
            'success': False,
            'message': f'Database error: {str(err)}'
        }), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()



@app.route('/api/forgot-password/reset', methods=['POST'])
def forgot_password():
    data = request.json
    email = data.get('email')
    dob = data.get('dob')
    mobile = data.get('mobile')
    new_password = data.get('newPassword')
    confirm_password = data.get('confirmPassword')
    role = data.get('role')

    # Validate passwords
    if new_password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match."}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Check user details in one query
        sql_verify = """
            SELECT * FROM users WHERE email=%s AND dob=%s AND mobile=%s AND role=%s
        """
        cursor.execute(sql_verify, (email, dob, mobile, role))
        user = cursor.fetchone()

        if not user:
            return jsonify({"success": False, "message": "Invalid details. User not found."}), 404

        # Update password
        sql_update = """
            UPDATE users SET password=%s WHERE email=%s AND role=%s
        """
        hashed_pw = hash_password(new_password)
        cursor.execute(sql_update, (hashed_pw, email, role))
        conn.commit()

        return jsonify({"success": True, "message": "Password changed successfully."})

    except mysql.connector.Error as err:
        return jsonify({"success": False, "message": f"Database error {str(err)}"}), 500

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
