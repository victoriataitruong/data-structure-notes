""" 
Array
"""
import array
my_array = array.array('i', [1, 2, 3, 4, 5])
print(my_array)





"""
ArrayList
"""
class ArrayList:
    def __init__(self):
        self.array = []

    def append(self, value):
        """Add an element to the end of the list."""
        self.array.append(value)

    def remove(self, value):
        """Remove the first occurrence of the element from the list."""
        if value in self.array:
            self.array.remove(value)
        else:
            print(f"{value} not found in the list.")

    def get(self, index):
        """Get the element at the given index."""
        if 0 <= index < len(self.array):
            return self.array[index]
        else:
            print("Index out of range.")
            return None

    def size(self):
        """Get the size of the list."""
        return len(self.array)

    def __str__(self):
        """Return a string representation of the list."""
        return str(self.array)
# Example usage:
arraylist = ArrayList()
arraylist.append(10)
arraylist.append(20)
arraylist.append(30)
print(arraylist)  
print(arraylist.get(1))  
arraylist.remove(20)
print(arraylist)  
print(arraylist.size()) 





"""
AVL Tree
"""
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # height of the node

class AVLTree:
    def insert(self, root, key):
        # 1. Perform the normal BST insertion
        if not root:
            return Node(key)

        if key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)

        # 2. Update the height of this ancestor node
        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))

        # 3. Get the balance factor and balance the tree
        balance = self.get_balance(root)

        # Left Left Case
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)

        # Right Right Case
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)

        # Left Right Case
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)

        # Right Left Case
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def left_rotate(self, z):
        y = z.right
        T2 = y.left

        # Perform rotation
        y.left = z
        z.right = T2

        # Update heights
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))

        return y

    def right_rotate(self, z):
        y = z.left
        T3 = y.right

        # Perform rotation
        y.right = z
        z.left = T3

        # Update heights
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))

        return y

    def get_height(self, root):
        if not root:
            return 0
        return root.height

    def get_balance(self, root):
        if not root:
            return 0
        return self.get_height(root.left) - self.get_height(root.right)

    def pre_order(self, root):
        if not root:
            return
        print(root.key, end=" ")
        self.pre_order(root.left)
        self.pre_order(root.right)


# Example usage
if __name__ == "__main__":
    tree = AVLTree()
    root = None

    keys = [10, 20, 30, 40, 50, 25]
    for key in keys:
        root = tree.insert(root, key)

    print("Preorder traversal of the AVL tree:")
    tree.pre_order(root)





"""
Binary Search Tree
"""
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.value = key


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node, key):
        if key < node.value:
            if node.left is None:
                node.left = Node(key)
            else:
                self._insert(node.left, key)
        elif key > node.value:
            if node.right is None:
                node.right = Node(key)
            else:
                self._insert(node.right, key)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None or node.value == key:
            return node
        if key < node.value:
            return self._search(node.left, key)
        return self._search(node.right, key)

    def inorder(self):
        return self._inorder(self.root)

    def _inorder(self, node):
        res = []
        if node:
            res = self._inorder(node.left)
            res.append(node.value)
            res = res + self._inorder(node.right)
        return res

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return node

        if key < node.value:
            node.left = self._delete(node.left, key)
        elif key > node.value:
            node.right = self._delete(node.right, key)
        else:
            # Node with one or no child
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            # Node with two children: get the inorder successor
            node.value = self._min_value_node(node.right).value
            node.right = self._delete(node.right, node.value)

        return node

    def _min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current


# Example usage:
bst = BinarySearchTree()
bst.insert(50)
bst.insert(30)
bst.insert(20)
bst.insert(40)
bst.insert(70)
bst.insert(60)
bst.insert(80)

print("Inorder traversal:", bst.inorder())  # Output: [20, 30, 40, 50, 60, 70, 80]

# Searching for a value
result = bst.search(40)
print("Search for 40:", "Found" if result else "Not found")

# Deleting a node
bst.delete(20)
print("Inorder traversal after deleting 20:", bst.inorder())





"""
Graph
"""
class Graph:
    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, node1, node2):
        if node1 not in self.graph:
            self.add_node(node1)
        if node2 not in self.graph:
            self.add_node(node2)
        self.graph[node1].append(node2)
        self.graph[node2].append(node1)  # For an undirected graph

    def print_graph(self):
        for node in self.graph:
            print(f"{node}: {self.graph[node]}")

# Example Usage
g = Graph()
g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(2, 4)
g.add_edge(3, 4)
g.print_graph()





"""
HashTable
"""
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [None] * size

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash(key)
        if self.table[index] is None:
            self.table[index] = [(key, value)]
        else:
            # If a key already exists, update its value
            for i, (k, v) in enumerate(self.table[index]):
                if k == key:
                    self.table[index][i] = (key, value)
                    return
            self.table[index].append((key, value))

    def get(self, key):
        index = self._hash(key)
        if self.table[index] is None:
            return None
        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def remove(self, key):
        index = self._hash(key)
        if self.table[index] is None:
            return None
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                del self.table[index][i]
                return
        return None

    def display(self):
        for i, bucket in enumerate(self.table):
            if bucket is not None:
                print(f"Index {i}: {bucket}")
            else:
                print(f"Index {i}: Empty")

# Example usage:
ht = HashTable()

ht.insert("name", "Alice")
ht.insert("age", 30)
ht.insert("city", "New York")

print(ht.get("name"))  # Output: Alice
print(ht.get("age"))   # Output: 30
print(ht.get("city"))  # Output: New York

ht.remove("age")
print(ht.get("age"))   # Output: None

ht.display()





"""
Heap
"""
class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, index):
        return (index - 1) // 2

    def left_child(self, index):
        return 2 * index + 1

    def right_child(self, index):
        return 2 * index + 2

    def heapify_up(self, index):
        while index > 0 and self.heap[self.parent(index)] > self.heap[index]:
            self.heap[self.parent(index)], self.heap[index] = self.heap[index], self.heap[self.parent(index)]
            index = self.parent(index)

    def heapify_down(self, index):
        smallest = index
        left = self.left_child(index)
        right = self.right_child(index)

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self.heapify_down(smallest)

    def insert(self, value):
        self.heap.append(value)
        self.heapify_up(len(self.heap) - 1)

    def extract_min(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()

        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify_down(0)
        return root

    def peek(self):
        return self.heap[0] if self.heap else None

    def is_empty(self):
        return len(self.heap) == 0

# Example usage:
heap = MinHeap()
heap.insert(10)
heap.insert(5)
heap.insert(20)
heap.insert(2)

print(heap.peek())  # Output: 2
print(heap.extract_min())  # Output: 2
print(heap.peek())  # Output: 5





"""
Linked List
"""
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    # Insert at the end of the linked list
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node
    
    # Insert at the beginning of the linked list
    def prepend(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    # Delete the first occurrence of a value in the linked list
    def delete(self, key):
        temp = self.head
        
        # If the key is the head node
        if temp and temp.data == key:
            self.head = temp.next
            temp = None
            return
        
        # Search for the key to be deleted
        prev = None
        while temp and temp.data != key:
            prev = temp
            temp = temp.next
        
        # If key was not found
        if not temp:
            return
        
        # Unlink the node from the linked list
        prev.next = temp.next
        temp = None
    
    # Display the linked list
    def display(self):
        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")
    
# Example usage:
linked_list = LinkedList()
linked_list.append(10)
linked_list.append(20)
linked_list.prepend(5)
linked_list.display()  # Output: 5 -> 10 -> 20 -> None
linked_list.delete(10)
linked_list.display()  # Output: 5 -> 20 -> None





"""
Lists
"""
# Example of a list
my_list = [1, 2, 3, 4, 5]
print(my_list)





"""
Map
"""
# Create a map using a dictionary
my_map = {}

# Add some key-value pairs
my_map["apple"] = 1
my_map["banana"] = 2
my_map["cherry"] = 3

# Access values by keys
print(my_map["apple"])   # Output: 1
print(my_map["banana"])  # Output: 2

# Check if a key exists
if "cherry" in my_map:
    print("Cherry is in the map!")

# Remove a key-value pair
del my_map["banana"]

# Print the updated map
print(my_map)  # Output: {'apple': 1, 'cherry': 3}





"""
Queue
"""
class Queue:
    def __init__(self):
        self.queue = []

    # Enqueue: Adds an item to the back of the queue
    def enqueue(self, item):
        self.queue.append(item)

    # Dequeue: Removes and returns the item from the front of the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            print("Queue is empty!")
            return None

    # Peek: Returns the front item without removing it
    def peek(self):
        if len(self.queue) > 0:
            return self.queue[0]
        else:
            print("Queue is empty!")
            return None

    # IsEmpty: Returns True if the queue is empty
    def is_empty(self):
        return len(self.queue) == 0

    # Size: Returns the number of items in the queue
    def size(self):
        return len(self.queue)

# Example usage
queue = Queue()
queue.enqueue(10)
queue.enqueue(20)
queue.enqueue(30)

print("Front of the queue:", queue.peek())  # Output: 10
print("Dequeue:", queue.dequeue())  # Output: 10
print("Queue size:", queue.size())  # Output: 2
print("Is the queue empty?", queue.is_empty())  # Output: False





"""
Set
"""
class MySet:
    def __init__(self):
        self.data = []
    
    def add(self, element):
        if element not in self.data:
            self.data.append(element)
    
    def remove(self, element):
        if element in self.data:
            self.data.remove(element)
    
    def contains(self, element):
        return element in self.data
    
    def size(self):
        return len(self.data)
    
    def __str__(self):
        return "{" + ", ".join(str(x) for x in self.data) + "}"
    

# Example usage:
my_set = MySet()
my_set.add(1)
my_set.add(2)
my_set.add(3)
print(my_set)  # Output: {1, 2, 3}

my_set.add(2)  # Duplicate, won't be added
print(my_set)  # Output: {1, 2, 3}

my_set.remove(2)
print(my_set)  # Output: {1, 3}

print(my_set.contains(1))  # Output: True
print(my_set.contains(2))  # Output: False





"""
Stack
"""
class Stack:
    def __init__(self):
        self.stack = []
    
    def is_empty(self):
        return len(self.stack) == 0
    
    def push(self, item):
        self.stack.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            raise IndexError("pop from empty stack")
    
    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            raise IndexError("peek from empty stack")
    
    def size(self):
        return len(self.stack)

# Example usage:
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # Output: 3
print(stack.peek())  # Output: 2
print(stack.size())  # Output: 2




"""
Trie
"""
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Example usage:
trie = Trie()
trie.insert("apple")
trie.insert("app")
print(trie.search("apple"))  # True
print(trie.search("app"))    # True
print(trie.search("appl"))   # False
print(trie.starts_with("ap")) # True





