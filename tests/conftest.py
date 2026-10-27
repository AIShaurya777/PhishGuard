# Import first: on macOS it limits OpenMP to one thread before torch or xgboost
# start theirs (see the note at the top of phishguard/deep.py).
import phishguard.deep  # noqa: F401
