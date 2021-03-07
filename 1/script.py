import numpy as np


print("Hello World")

a = np.array([[2,3,4],[1,5,8]], dtype=np.int32)   # create array
b = np.array([[3,2,9],[2,6,5]], dtype=np.int32)   # create array

print("The 'a' array:\n", a)
print("The 'b' array:\n", b)
print("Sum of the two arrays, elementwise:\n", a + b)
print("Multiplying all elements of 'a' by 2:\n", 2*a)
print("Adding up all elements of array 'a'. The sum is:", np.sum(a))


a = np.array([[2,3,4],[1,5,8]], dtype=np.int32)

print("The shape of the array:", a.shape)   # a tuple
print("The data type of the array:", a.dtype)   # the type of the elements
print("The number of axes (dimensions of the array):", a.ndim)   # a.k.a length of the shape tuple
print("The size of the array (total number of elements):", a.size)   # the product of the shape
print("Length along the first axis:", a.shape[0])
print("Length along the first axis:", len(a))   # len() built-in python function

print("The strides of the array", a.strides)  # tells us how many bytes do we need to move in
                                              #   the memory to increase index along each axis
  


a = np.arange(6, dtype=np.int32)
print("The 'a' array:\n", a)
print("   ... its shape is", a.shape)   # a tuple

b = a.reshape((2, 3))   # reshape to 2D shape: (2, 3)
print("\nThe 'a' array reshaped to 2D shape (2, 3):\n", b)
print("   ... its shape is", b.shape)

# We modify an item in the original array
a[0] = 42
print("\nThe modified 'a' array:\n", a)
print("The 2D 'b' view of the 'a' array is also modified:\n", b)

# Now we modify an item in the view array
b[1,1] = 99
print("\nThe content of the 'a' array after 2nd modification:\n", a)
print("The 'b' view after 2nd modification:", b)