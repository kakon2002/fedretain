Smoke tests for the legacy scripts. CPU only, synthetic data, about 10 minutes in total. Run in this order
from any directory (test_fix.py creates the synthetic data that the other two reuse):

    python legacy/tests/test_fix.py
    python legacy/tests/test_extra.py
    python legacy/tests/test_new.py

Scratch data goes to legacy/tests/_synthetic (override with FEDFREQ_TEST_DIR). Needs: torch, torchvision,
numpy, pandas, pillow, scikit-learn, matplotlib.
