import cv2


class HITLInterface:
    def __init__(self):
        self.color_box = (0, 255, 0)
        self.color_text = (255, 255, 255)
        self.color_valid = (0, 255, 0)        # Hijau - tepat target
        self.color_hampir = (0, 255, 255)     # Kuning - selisih 1
        self.color_warning = (0, 0, 255)      # Merah - selisih >= 2

    def draw(self, frame, boxes, fps, scenario_name, target_count):
        """
        Menggambar UI di atas frame.
        target_count == 0 -> mode BEBAS (tanpa freeze, cuma info + tombol S/Q)
        target_count > 0  -> mode TARGET (freeze saat pas, warna sesuai selisih)
        Return: (frame_hasil, trigger_freeze)
        """
        h, w = frame.shape[:2]
        detected_count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2, score = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_box, 2)
            cv2.putText(frame, f"{score:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_box, 1)

        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

        if target_count == 0:
            # --- MODE BEBAS: tidak ada target, tidak ada freeze ---
            info_text = f"MODE: {scenario_name} | FPS: {fps:.1f} | Deteksi: {detected_count} (Target: Bebas)"
            cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_text, 2)
            cv2.putText(frame, "[1]/[2] Ganti Model  |  [S] Simpan  |  [Q] Keluar",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.color_text, 1)
            return frame, False

        # --- MODE TARGET: freeze + warna sesuai selisih ---
        info_text = f"MODE: {scenario_name} | FPS: {fps:.1f} | Deteksi: {detected_count} / Target: {target_count}"
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_text, 2)

        diff = detected_count - target_count
        trigger_freeze = False

        if diff == 0:
            cv2.rectangle(frame, (0, 0), (w, h), self.color_valid, 10)
            cv2.putText(frame, "VALID! TARGET TERCAPAI", (w // 2 - 220, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_valid, 3)
            cv2.putText(frame, "[Y] SIMPAN & LANJUT  |  [N] ULANGI", (w // 2 - 250, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_valid, 2)
            trigger_freeze = True

        elif abs(diff) == 1:
            arah = "KURANG" if diff < 0 else "LEBIH"
            cv2.rectangle(frame, (0, 0), (w, h), self.color_hampir, 10)
            cv2.putText(frame, f"HAMPIR! {arah} 1 PART", (w // 2 - 200, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color_hampir, 3)

        elif diff < 0:
            cv2.rectangle(frame, (0, 0), (w, h), self.color_warning, 10)
            cv2.putText(frame, f"PART KURANG! ({abs(diff)} lagi)", (w // 2 - 250, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_warning, 3)

        else:
            cv2.rectangle(frame, (0, 0), (w, h), self.color_warning, 10)
            cv2.putText(frame, f"PART BERLEBIH! (+{diff})", (w // 2 - 250, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_warning, 3)

        if not trigger_freeze:
            cv2.putText(frame, "[1]/[2] Ganti Model  |  [Q] Keluar",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.color_text, 1)

        return frame, trigger_freeze