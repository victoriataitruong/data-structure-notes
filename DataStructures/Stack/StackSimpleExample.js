class Stack {
  constructor() {
    this.items = [];
  }

  // Add item to the top of the stack
  push(item) {
    this.items.push(item);
  }

  // Remove and return the item from the top of the stack
  pop() {
    if (this.isEmpty()) return "Stack is empty";
    return this.items.pop();
  }

  // View the item at the top without removing it
  peek() {
    return this.items[this.items.length - 1];
  }

  // Check if the stack is empty
  isEmpty() {
    return this.items.length === 0;
  }
}

// Create a stack
let stack = new Stack();
stack.push("Toy 1");
stack.push("Toy 2");
stack.push("Toy 3");

console.log(stack.pop()); // Removes "Toy 3"
console.log(stack.peek()); // Shows "Toy 2" (top of the stack)
