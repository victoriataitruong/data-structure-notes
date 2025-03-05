// This is how you create a node in a linked list
class Node {
  constructor(data) {
    this.data = data; // The treasure (data) the clue holds
    this.next = null; // The next clue (next node) in the list
  }
}

// Let's make a linked list now
class LinkedList {
  constructor() {
    this.head = null; // The starting point of the linked list (the first clue)
  }

  // Add a new node at the end of the list
  append(data) {
    const newNode = new Node(data); // Create a new node (new clue)
    if (this.head === null) {
      this.head = newNode; // If the list is empty, this is the first clue
    } else {
      let current = this.head;
      while (current.next !== null) {
        current = current.next; // Find the last clue
      }
      current.next = newNode; // Link the last clue to the new clue
    }
  }

  // Print all the clues in the list
  print() {
    let current = this.head;
    while (current !== null) {
      console.log(current.data); // Print the treasure (data) in the clue
      current = current.next; // Move to the next clue
    }
  }
}

// Let's create a linked list and add clues to it
let myList = new LinkedList();
myList.append("Clue 1: Go to the park.");
myList.append("Clue 2: Look under the big tree.");
myList.append("Clue 3: Check the mailbox.");

myList.print(); // Output the clues in the linked list
