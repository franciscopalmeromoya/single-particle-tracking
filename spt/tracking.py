import numpy as np
import pandas as pd
from .core import run

class Tracker:
    """
    Class for tracking spots between frames.

    Parameters
    ----------
    skip_frames : int
        Max number of frames to skip for segment linking.
    max_dist : float
        Maximum distance between points for frame-to-frame linking (pixels).
    """
    def __init__(self, skip_frames : int = 1, max_dist : float = 10) -> None:
        self.skip_frames = skip_frames
        self.max_dist = max_dist

    @staticmethod
    def init(df_tracks : pd.DataFrame):
        spots = np.zeros((len(df_tracks), 4))
        spots[:, 0] = df_tracks["frame"].values
        spots[:, 1] = df_tracks["fidx"].values    
        spots[:, 2] = df_tracks["x"].values
        spots[:, 3] = df_tracks["y"].values  
        return spots
    
    @staticmethod
    def back(dfspots : np.ndarray):
        df_tracks = pd.DataFrame({
            "frame" : dfspots[:, 0].astype(int),
            "fidx" : dfspots[:, 1].astype(int),
            "x" : dfspots[:, 2],
            "y" : dfspots[:, 3],
            "tid" : dfspots[:, 4].astype(int)
        }, index=dfspots[:, 5].astype(int))
        return df_tracks

    def track(self, df_tracks : pd.DataFrame):
        """Do LAP tracking. df_tracks_input needs to be a dataframe with columns ['frame', 'x', 'y'];
        output will contain a new column 'track_id'.

        Parameters
        ----------
        df_tracks : pandas DataFrame
            Input dataframe with columns ['frame', 'frame_subindex', 'x', 'y'].

        Returns
        -------
        df_tracks : pandas DataFrame
            Output dataframe - same as input, but with added column 'track_id'.
        """
        # Convert to spots array
        spots = self.init(df_tracks)

        # Start
        dfspots = run(spots, self.skip_frames, self.max_dist)
        
        # Convert back to dataframe
        return self.back(dfspots)