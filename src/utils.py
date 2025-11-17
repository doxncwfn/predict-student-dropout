import os
import matplotlib.pyplot as plt

def get_project_paths():
    PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "../data")
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
    os.makedirs(IMAGES_PATH, exist_ok=True)
    return PROJECT_ROOT_DIR, DATA_PATH, IMAGES_PATH

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300, show=True):
    _, _, IMAGES_PATH = get_project_paths()
    path = os.path.join(IMAGES_PATH, f"{fig_id}.{fig_extension}")
    print(f"Saving figure {fig_id} at {path}")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    if show:
        plt.show()