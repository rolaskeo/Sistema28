import os, sys, subprocess

def main():
    proj = r"C:\Rolex\Python\Sistema28Script"
    script = os.path.join(proj, "streamlit_carga.py")
    # Lanza streamlit con tu python actual, en el directorio del proyecto
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", script, "--server.port", "8504"], cwd=proj)

if __name__ == "__main__":
    main()
