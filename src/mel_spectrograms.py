import cv2
import os
import pandas as pd
import gc


def loop_files():
    df = pd.DataFrame(columns=["name", "genre", "spectrogram"])
    for subdir, _, files in os.walk("pngs"):
        for file in files:
            genre = subdir[len("pngs") + 1:]
            img = cv2.imread(f"pngs/{genre}/{file}", 0)
            cropped_img = img[35:35 + 217, 55:335 + 55]
            # New img size is 217x335

            df = df.append(
                {
                    "name": file,
                    "genre": genre,
                    "spectrogram": cropped_img,
                },
                ignore_index=True,
            )
            print(f"/{genre}/{file}")
            gc.collect()

    df.to_pickle("mel_spectrograms.pkl")
