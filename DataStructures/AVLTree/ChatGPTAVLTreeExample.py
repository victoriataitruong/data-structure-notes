class TreeNode:
    def __init__(self, val):
        self.val = val  # Value of the node
        self.left = None  # Left child of the node
        self.right = None  # Right child of the node
        self.height = 1  # Initially, the height of a new node is 1

class AVLTree:
    def height(self, node):
        return node.height if node else 0

    def get_balance_factor(self, node):
        return self.height(node.left) - self.height(node.right) if node else 0

    def right_rotate(self, y):
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        y.height = max(self.height(y.left), self.height(y.right)) + 1
        x.height = max(self.height(x.left), self.height(x.right)) + 1

        return x

    def left_rotate(self, x):
        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        x.height = max(self.height(x.left), self.height(x.right)) + 1
        y.height = max(self.height(y.left), self.height(y.right)) + 1

        return y

    def left_right_rotate(self, node):
        node.left = self.left_rotate(node.left)
        return self.right_rotate(node)

    def right_left_rotate(self, node):
        node.right = self.right_rotate(node.right)
        return self.left_rotate(node)

    def insert(self, node, val):
        if not node:
            return TreeNode(val)

        if val < node.val:
            node.left = self.insert(node.left, val)
        elif val > node.val:
            node.right = self.insert(node.right, val)
        else:
            return node

        node.height = 1 + max(self.height(node.left), self.height(node.right))
        balance = self.get_balance_factor(node)

        if balance > 1 and val < node.left.val:
            return self.right_rotate(node)
        if balance < -1 and val > node.right.val:
            return self.left_rotate(node)
        if balance > 1 and val > node.left.val:
            return self.left_right_rotate(node)
        if balance < -1 and val < node.right.val:
            return self.right_left_rotate(node)

        return node

    def inorder_traversal(self, root):
        if root:
            self.inorder_traversal(root.left)
            print(root.val, end=' ')
            self.inorder_traversal(root.right)

if __name__ == "__main__":
    tree = AVLTree()
    root = None

    root = tree.insert(root, 10)
    root = tree.insert(root, 20)
    root = tree.insert(root, 30)
    root = tree.insert(root, 15)

    print("Inorder traversal of the AVL tree:")
    tree.inorder_traversal(root)  # Expected output: 10 15 20 30
