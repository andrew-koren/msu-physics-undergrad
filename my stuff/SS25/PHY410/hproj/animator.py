from IPython.display import HTML
import os
import base64


class ParticleAnimator:
    def __init__(self, fig, Writer, filename):
        self.fig = fig
        self.Writer = Writer
        self.filename = filename
        # List of animation tasks.
        self.animation_sequence = []
        # Optionally hold registered artists (if needed across tasks).
        self.artists = {}

    def register_artist(self, name, artist): #optional
        self.artists[name] = artist

    def set_display(self, fps, total_time):
        """Set up simulation display parameters."""
        self.fps = fps
        self.T = total_time
        self.dt = 1 / fps
        self.frames = int(total_time * fps)

    def generate_frame_data(self, data, update_period=None, update_fps=None):
        """
        Given a list 'data' (each element holding frame input) and an update condition,
        this function creates a dictionary mapping each simulation frame (0-indexed). Will truncate extra data
        """
        if update_fps is not None:
            update_interval = self.fps / update_fps
        elif update_period is not None:
            update_interval = update_period / self.dt
        else:
            update_interval = 1  # Update every frame

        # Create a dictionary of frame data. At frames that are multiples of update_interval,
        # we pick the next value from 'data' (if available). Otherwise, repeat the last value.
        frame_data = {}
        current_data = None
        update_index = 0
        for i in range(self.frames):
            if i % update_interval < 1 and update_index < len(data):
                current_data = data[update_index]
                update_index += 1
            frame_data[i] = current_data
        return frame_data

    def animate_ax(self, ax, method, frame_data, *args):
        """
        Register an animation task. 
        - ax: the axis to update.
        - method: update function that takes (ax, frame_data, *args)
        - frame_data: a dictionary mapping frame number to data.
        - args: additional fixed arguments (e.g., dynamic artist handles) to pass.
        """
        self.animation_sequence.append({
            'ax': ax,
            'method': method,
            'frame_data': frame_data,
            'args': args
        })

    def make_animation(self):
        """
        Loop over frames, calling each animation task's method if its update data exists,
        then capturing the frame with writer.grab_frame().
        """
        writer = self.Writer(self.fps)
        with writer.saving(self.fig, self.filename, dpi=100):
            for i in range(self.frames):
                for anim in self.animation_sequence:
                    frame_input = anim['frame_data'][i]
                    if frame_input is not None:
                        anim['method'](anim['ax'], frame_input, *anim['args'])
                writer.grab_frame()
    
    def display(self, width=800):
        """Auto-display an animation inline (MP4, GIF, or WebP) in a Jupyter notebook."""
        ext = os.path.splitext(self.filename)[1].lower()

        with open(self.filename, "rb") as file_:
            data = file_.read()
        encoded = base64.b64encode(data).decode("utf-8")

        # ffmpeg or pillow
        if ext == ".mp4":
            return HTML(f"""
            <video width="{width}" controls>
                <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """)
        elif ext == ".gif":
            return HTML(f"""
            <img src="data:image/gif;base64,{encoded}" width="{width}">
            """)
        elif ext == ".webp":
            return HTML(f"""
            <img src="data:image/webp;base64,{encoded}" width="{width}">
            """)
        else:
            return HTML("<p>Unsupported file format for inline display.</p>")