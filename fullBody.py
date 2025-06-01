import cv2

def main():
   
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

    if cascade.empty():
        print("Error al cargar.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Detección de Cuerpo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detección de Cuerpo", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        bodies = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(100, 100)
        )

        if len(bodies) > 0:
            x, y, w, h = max(bodies, key=lambda b: b[2] * b[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, "Full Body", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        cv2.imshow("Detección de Cuerpo", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Salir con ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
