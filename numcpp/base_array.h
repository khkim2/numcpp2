#ifndef NUMCPP_BASE_ARRAY_H_
#define NUMCPP_BASE_ARRAY_H_

#include "tuple.h"

#include <memory>
#include <assert.h>

namespace np {

struct heap_allocator
{
	static void *allocate(int size)
	{
		return new char[size];
	}

	static void free(void *ptr)
	{
		delete[] reinterpret_cast<char *>(ptr);
	}
};

struct base_array_t
{
//private:
public:
	const int _itemSize;
	tuple _size;
	std::unique_ptr<int[]> _stride;
	std::shared_ptr<void> _address;
	void *_origin;

public:
	base_array_t() : 
		_itemSize(1), _origin(nullptr)
	{
	}

	base_array_t(int itemSize) : 
		_itemSize(itemSize), _origin(nullptr) 
	{ 
	}

	// copy constructor
	base_array_t(const base_array_t &other) : 
		_itemSize(other._itemSize), 
		_size(other._size), 
		_stride(new int[other.ndims()])
	{
		std::copy(other._stride.get(), other._stride.get() + ndims(), _stride.get());

		_address = other._address; // add reference
		_origin = other._origin;
	}

	// copy assign
	const base_array_t &operator=(const base_array_t &other)
	{
		// TODO: assign _itemSize

		_size = other._size;
		_stride.reset(new int[other.ndims()]);
		std::copy(other._stride.get(), other._stride.get() + ndims(), _stride.get());

		_address = other._address; // add reference
		_origin = other._origin;

		return *this;
	}

	// move constructor
	base_array_t(base_array_t &&other) : 
		_itemSize(other._itemSize), 
		_size(std::move(other._size)), 
		_stride(std::move(other._stride)), 
		_address(other._address), 
		_origin(other._origin)
	{
		other._origin = nullptr;
	}

	// move assign
	const base_array_t &operator=(base_array_t &&other)
	{
		_size = std::move(other._size);
		_stride = std::move(other._stride);
		_address = std::move(other._address);
		_origin = other._origin;

		other._origin = nullptr;

		return *this;
	}

private:
	void init(
		int ndims, int *shape, 
		std::shared_ptr<void> address, void *origin)
	{
		_size = tuple(ndims, shape);
		_stride.reset(new int[ndims]);
		_stride[0] = itemSize();
		for (int i = 1; i < ndims; i++)
			_stride[i] = _stride[i - 1] * shape[i - 1];

		_address = address;
		_origin = origin;
	}

	template <class Allocator>
	void init(int ndims, int size, int *shape)
	{
		void *ptr = Allocator::allocate(size * _itemSize);
		auto address = std::shared_ptr<void>(ptr, Allocator::free);

		init(ndims, shape, address, ptr);
	}

public:
	template <class Allocator>
	void setSize(int size0)
	{
		if (this->ndims() == 1 && 
			this->size(0) == size0) return;

		int shape[1] = { size0 };
		init<Allocator>(1, size0, shape);
	}

	template <class Allocator>
	void setSize(int size0, int size1) {
		if (this->ndims() == 2 && 
			this->size(0) == size0 && 
			this->size(1) == size1) return;

		int shape[2] = { size0, size1 };
		init<Allocator>(2, size0 * size1, shape);
	}

	template <class Allocator>
	void setSize(int size0, int size1, int size2)
	{
		if (this->ndims() == 3 && 
			this->size(0) == size0 && 
			this->size(1) == size1 && 
			this->size(2) == size2) return;

		int shape[3] = { size0, size1, size2 };
		init<Allocator>(3, size0 * size1 * size2, shape);
	}

	template <class Allocator>
	void setSize(int ndims, int *shape)
	{
		if (this->ndims() == ndims)
		{
			for (int i = 0; i < ndims; i++)
				if (shape[i] != this->size(i))
					goto allocate;

			return;
		}

allocate:
		int size = 1;
		for (int i = 0; i < ndims; i++)
			size *= shape[i];

		init<Allocator>(ndims, size, shape);
	}

	base_array_t slice(int from, int to)
	{
		assert(from <= to);	

		base_array_t result(itemSize());

		int *shape = new int[1];
		shape[0] = to - from;
		result._size = tuple(1, shape);

		int *stride = new int[1];
		stride[0] = this->stride(0);
		result._stride = std::unique_ptr<int[]>(stride);

		// add reference count here
		result._address = this->_address;

		// new origin with offset
		result._origin = this->ptr_at(from);

		return result;
	}

	base_array_t slice(int from0, int from1, int to0, int to1)
	{
		assert(from0 <= to0);	
		assert(from1 <= to1);	

		base_array_t result(itemSize());

		int *shape = new int[2];
		shape[0] = to0 - from0;
		shape[1] = to1 - from1;
		result._size = tuple(2, shape);

		int *stride = new int[2];
		stride[0] = this->stride(0);
		stride[1] = this->stride(1);
		result._stride = std::unique_ptr<int[]>(stride);

		// add reference count here
		result._address = this->_address;

		// new origin with offset
		result._origin = this->ptr_at(from0, from1);

		return result;
	}

	// ## Access to premitive properties

	int itemSize() const
	{
		return _itemSize;
	}

	int ndims() const
	{
		return _size.size();
	}

	int *shape() const
	{
		return _size.ptr();
	}

	// TODO: Remove this function
	int shape(int dim) const
	{
		return _size[dim];
	}

	int size(int dim) const
	{
		return _size[dim];
	}

	int stride(int dim) const
	{
		return _stride[dim];
	}

	void *raw_ptr()
	{
		return _origin;
	}

	const void *raw_ptr() const
	{
		return _origin;
	}

	// ## Derived property functions

	bool empty() const
	{
		return raw_ptr() == nullptr || size() == 0;
	}

	// TODO: Rename to length()
	int size() const
	{
		int result = 1;
		for (int i = 0; i < ndims(); i++)
			result *= size(i);

		return result;
	}

	int byteSize() const
	{
		return size() * itemSize();
	}

	template <typename T>
	T *raw_ptr()
	{
		return static_cast<T *>(raw_ptr());
	}

	template <typename T>
	const T *raw_ptr() const
	{
		return static_cast<const T *>(raw_ptr());
	}

	void *ptr_at(int index0)
	{
		return raw_ptr<char>() + index0 * stride(0);
	}

	void *ptr_at(int index0, int index1)
	{
		return raw_ptr<char>() + index0 * stride(0) + index1 * stride(1);
	}

	void *ptr_at(int index0, int index1, int index2)
	{
		return raw_ptr<char>() + index0 * stride(0) + index1 * stride(1) + index2 * stride(2);
	}

	const void *ptr_at(int index0) const
	{
		return raw_ptr<char>() + index0 * stride(0);
	}

	const void *ptr_at(int index0, int index1) const
	{
		return raw_ptr<char>() + index0 * stride(0) + index1 * stride(1);
	}

	const void *ptr_at(int index0, int index1, int index2) const
	{
		return raw_ptr<char>() + index0 * stride(0) + index1 * stride(1) + index2 * stride(2);
	}

	template <typename T>
	T& at(int index0)
	{
		return *static_cast<T *>(ptr_at(index0));
	}

	template <typename T>
	T& at(int index0, int index1)
	{
		return *static_cast<T *>(ptr_at(index0, index1));
	}

	template <typename T>
	T& at(int index0, int index1, int index2)
	{
		return *static_cast<T *>(ptr_at(index0, index1, index2));
	}

	template <typename T>
	const T& at(int index0) const
	{
		return *static_cast<const T *>(ptr_at(index0));
	}

	template <typename T>
	const T& at(int index0, int index1) const
	{
		return *static_cast<const T *>(ptr_at(index0, index1));
	}

	template <typename T>
	const T& at(int index0, int index1, int index2) const
	{
		return *static_cast<const T *>(ptr_at(index0, index1, index2));
	}
};

} // namespace np

#endif // NUMCPP_ABSTRACT_ARRAY_H_