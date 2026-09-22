# Developing
The np.uint8, np.int32, np.bool_ on the arrays in World are a separate, deliberate choice: they set the storage type for the whole array (e.g. genome bits stored as 1 byte each instead of Python's ~28-byte int objects), and that's what makes the arrays cheap to store and fast to operate on — and it's a precondition for Numba later.

3