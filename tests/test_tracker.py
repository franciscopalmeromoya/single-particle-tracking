import spt
import pandas as pd
def test_tracker():

    tracking_data = {
        'frame':            [0, 1, 4, 5, 6,    0, 1, 2, 3, 4],
        'fidx':   [0, 0, 0, 0, 0,    1, 1, 0, 0, 1],
        'x':                [0, 0, 0, 0, 0,    10, 11, 12, 15, 16],
        'y':                [0, 0, 0, 0, 0,    0, 0, 0, 0, 0]
    }
    df_in = pd.DataFrame(tracking_data)

    tracker0 = spt.tracking.Tracker(skip_frames=3, max_dist=2)
    tracker1 = spt.tracking.Tracker(skip_frames=1, max_dist=5)

    df_out = tracker0.track(df_in)  # skip_frames=3 should put first five points together in a track.
    df0 = df_out[df_out['tid'] == 0]
    df1 = df_out[df_out['tid'] == 1]
    df2 = df_out[df_out['tid'] == 2]
    assert len(df0) == 5
    assert len(df1) == 3
    assert len(df2) == 2

    df_out = tracker1.track(df_in)  # max_dist=5 should put last five points together in a track.
    df0 = df_out[df_out['tid'] == 0]
    df1 = df_out[df_out['tid'] == 1]
    df2 = df_out[df_out['tid'] == 2]
    assert len(df0) == 2
    assert len(df1) == 5
    assert len(df2) == 3