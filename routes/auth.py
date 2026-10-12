from flask import (Blueprint, render_template, request,
                   redirect, url_for, session, flash)
from database.auth import AuthManager

auth_bp     = Blueprint('auth', __name__)
auth_manager = AuthManager()


def _logged_in():
    return 'user' in session


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if _logged_in():
        return redirect(url_for('main.dashboard'))

    error = None
    if request.method == 'POST':
        email    = request.form.get('email', '')
        password = request.form.get('password', '')
        ok, result = auth_manager.login(email, password)
        if ok:
            session.permanent = True
            session['user']   = result
            return redirect(url_for('main.dashboard'))
        error = result

    return render_template('auth/login.html', error=error)


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if _logged_in():
        return redirect(url_for('main.dashboard'))

    error   = None
    success = None
    if request.method == 'POST':
        name     = request.form.get('name', '')
        email    = request.form.get('email', '')
        password = request.form.get('password', '')
        ok, msg  = auth_manager.signup(name, email, password)
        if ok:
            success = msg
        else:
            error = msg

    return render_template('auth/signup.html', error=error, success=success)


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))
