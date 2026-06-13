from growthlens import create_app

app = create_app()

if __name__ == '__main__':
    # Running in debug mode for development ease
    app.run(host='127.0.0.1', port=5000, debug=True)
