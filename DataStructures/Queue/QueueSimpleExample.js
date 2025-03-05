class Queue {
  constructor() {
    this.items = [];
  }

  // Add item to the back of the queue
  enqueue(item) {
    this.items.push(item);
  }

  // Remove and return the item from the front of the queue
  dequeue() {
    if (this.isEmpty()) return "Queue is empty";
    return this.items.shift();
  }

  // View the item at the front without removing it
  peek() {
    return this.items[0];
  }

  // Check if the queue is empty
  isEmpty() {
    return this.items.length === 0;
  }
}

// Create a queue
let queue = new Queue();
queue.enqueue("Person 1");
queue.enqueue("Person 2");
queue.enqueue("Person 3");

console.log(queue.dequeue()); // Removes "Person 1"
console.log(queue.peek()); // Shows "Person 2" (front of the queue)
