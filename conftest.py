"""Present so pytest puts the project root on ``sys.path``.

That makes ``import app`` work from the test suite without installing the
package or setting PYTHONPATH.
"""
