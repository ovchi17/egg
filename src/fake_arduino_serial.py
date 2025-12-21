# macOS/Linux: creates a virtual serial port and streams 3-channel data like "v1,v2,v3\n"

import os
import pty
import time
import math
import random
import argparse


def gen_sample(t: float):
    """
    3 канала:
    - ch1: λ-подобный 5 Гц + шум
    - ch2: α-подобный 10 Гц + шум
    - ch3: медленная волна 1 Гц + шум
    """
    noise1 = random.gauss(0, 0.15)
    noise2 = random.gauss(0, 0.12)
    noise3 = random.gauss(0, 0.10)

    ch1 = 1.0 * math.sin(2 * math.pi * 5.0 * t) + noise1
    ch2 = 0.8 * math.sin(2 * math.pi * 10.0 * t + 0.7) + noise2
    ch3 = 0.6 * math.sin(2 * math.pi * 1.0 * t) + noise3

    return ch1, ch2, ch3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=float, default=250.0, help="частота дискретизации (Гц)")
    ap.add_argument("--delimiter", type=str, default=",", help="разделитель значений")
    args = ap.parse_args()

    master_fd, slave_fd = pty.openpty()
    slave_name = os.ttyname(slave_fd)

    print("\nВиртуальный serial создан!")
    print("Подключайся в твоём приложении к порту:")
    print(f"   {slave_name}\n")
    print("Формат строк: ch1,ch2,ch3\\n")
    print("Остановить: Ctrl+C\n")

    dt = 1.0 / args.fs
    t0 = time.perf_counter()
    next_t = t0

    try:
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0, next_t - now))

            t = time.perf_counter() - t0
            ch1, ch2, ch3 = gen_sample(t)

            line = f"{ch1:.6f}{args.delimiter}{ch2:.6f}{args.delimiter}{ch3:.6f}\n"
            os.write(master_fd, line.encode("utf-8"))

            next_t += dt

    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.close(master_fd)
        except Exception:
            pass
        try:
            os.close(slave_fd)
        except Exception:
            pass
        print("\nОстановлено.")


if __name__ == "__main__":
    main()