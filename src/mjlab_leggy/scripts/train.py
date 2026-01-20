"""Train script that loads mjlab_leggy tasks before running mjlab's training."""

# Import our tasks to register them with gymnasium
import mjlab_leggy.tasks  # noqa: F401

# Then run mjlab's main training function
from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
