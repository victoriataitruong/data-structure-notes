// Definition for a binary tree node.
class TreeNode {
    int val; // Value of the node
    TreeNode left; // Left child of the node
    TreeNode right; // Right child of the node
    int height; // Height of the node to maintain balance

    // Constructor to create a new node
    TreeNode(int val) {
        this.val = val; // Set the value of the node
        this.height = 1; // Initially, the height of a new node is 1
    }
}

public class AVLTree {

    // Function to get the height of a node (used to calculate balance factor)
    private int height(TreeNode node) {
        if (node == null) return 0; // Return 0 for null nodes
        return node.height; // Return the height of the node
    }

    // Function to get the balance factor of a node
    private int getBalanceFactor(TreeNode node) {
        if (node == null) return 0; // Return 0 for null nodes
        // Balance factor = height of left subtree - height of right subtree
        return height(node.left) - height(node.right);
    }

    // Right rotation to fix imbalance (Right-Right case)
    private TreeNode rightRotate(TreeNode y) {
        TreeNode x = y.left; // Set x as the left child of y
        TreeNode T2 = x.right; // Set T2 as the right child of x

        // Perform rotation
        x.right = y; // Make y the right child of x
        y.left = T2; // Set T2 as the left child of y

        // Update heights of y and x
        y.height = Math.max(height(y.left), height(y.right)) + 1;
        x.height = Math.max(height(x.left), height(x.right)) + 1;

        return x; // Return new root (x)
    }

    // Left rotation to fix imbalance (Left-Left case)
    private TreeNode leftRotate(TreeNode x) {
        TreeNode y = x.right; // Set y as the right child of x
        TreeNode T2 = y.left; // Set T2 as the left child of y

        // Perform rotation
        y.left = x; // Make x the left child of y
        x.right = T2; // Set T2 as the right child of x

        // Update heights of x and y
        x.height = Math.max(height(x.left), height(x.right)) + 1;
        y.height = Math.max(height(y.left), height(y.right)) + 1;

        return y; // Return new root (y)
    }

    // Left-Right rotation (double rotation) to fix imbalance
    private TreeNode leftRightRotate(TreeNode node) {
        node.left = leftRotate(node.left); // First, perform left rotation on the left child
        return rightRotate(node); // Then, perform right rotation
    }

    // Right-Left rotation (double rotation) to fix imbalance
    private TreeNode rightLeftRotate(TreeNode node) {
        node.right = rightRotate(node.right); // First, perform right rotation on the right child
        return leftRotate(node); // Then, perform left rotation
    }

    // Function to insert a node into the AVL tree
    public TreeNode insert(TreeNode node, int val) {
        // Step 1: Perform the normal BST insert
        if (node == null) return new TreeNode(val); // If the node is null, create a new node

        // Insert the node like a regular BST
        if (val < node.val) {
            node.left = insert(node.left, val); // Insert in the left subtree
        } else if (val > node.val) {
            node.right = insert(node.right, val); // Insert in the right subtree
        } else {
            // Duplicate values are not allowed in the AVL tree
            return node; // Return unchanged node if value already exists
        }

        // Step 2: Update the height of the current node
        node.height = 1 + Math.max(height(node.left), height(node.right)); // Set the height of the node

        // Step 3: Get the balance factor of the current node
        int balance = getBalanceFactor(node); // Calculate balance factor

        // If the node becomes unbalanced, perform the appropriate rotations

        // Left-Left case: Right rotation
        if (balance > 1 && val < node.left.val) {
            return rightRotate(node); // Perform right rotation
        }

        // Right-Right case: Left rotation
        if (balance < -1 && val > node.right.val) {
            return leftRotate(node); // Perform left rotation
        }

        // Left-Right case: Left rotation followed by right rotation
        if (balance > 1 && val > node.left.val) {
            return leftRightRotate(node); // Perform left-right rotation
        }

        // Right-Left case: Right rotation followed by left rotation
        if (balance < -1 && val < node.right.val) {
            return rightLeftRotate(node); // Perform right-left rotation
        }

        // Return the (unchanged) node pointer
        return node;
    }

    // Function to print the inorder traversal of the AVL tree (for testing)
    public void inorderTraversal(TreeNode root) {
        if (root != null) {
            inorderTraversal(root.left); // Traverse the left subtree
            System.out.print(root.val + " "); // Print the node's value
            inorderTraversal(root.right); // Traverse the right subtree
        }
    }

    public static void main(String[] args) {
        AVLTree tree = new AVLTree(); // Create an instance of AVLTree
        TreeNode root = null; // Initialize the root as null

        // Insert nodes into the AVL tree
        root = tree.insert(root, 10); // Insert node with value 10
        root = tree.insert(root, 20); // Insert node with value 20
        root = tree.insert(root, 30); // Insert node with value 30
        root = tree.insert(root, 15); // Insert node with value 15 (triggers Left-Right rotation)

        // Print inorder traversal of the AVL tree
        System.out.println("Inorder traversal of the AVL tree:");
        tree.inorderTraversal(root);  // Should print: 10 15 20 30
    }
}