class TrieNode {
  constructor() {
    this.children = {}; // Holds child nodes
    this.isEndOfWord = false; // Marks the end of a word
  }
}

class Trie {
  constructor() {
    this.root = new TrieNode(); // The root node is empty
  }

  // Insert a word into the Trie
  insert(word) {
    let currentNode = this.root; // Start at the root node

    for (let char of word) {
      if (!currentNode.children[char]) {
        currentNode.children[char] = new TrieNode(); // Create a new node if it doesn't exist
      }
      currentNode = currentNode.children[char]; // Move to the next node
    }

    currentNode.isEndOfWord = true; // Mark the end of the word
  }

  // Search for a word in the Trie
  search(word) {
    let currentNode = this.root;

    for (let char of word) {
      if (!currentNode.children[char]) {
        return false; // Word not found
      }
      currentNode = currentNode.children[char]; // Move to the next node
    }

    return currentNode.isEndOfWord; // Check if it's the end of the word
  }
}

// Example usage
let trie = new Trie();
trie.insert("cat");
trie.insert("dog");

console.log(trie.search("cat")); // true
console.log(trie.search("dog")); // true
console.log(trie.search("bat")); // false
