class PriorityQueue {
    constructor() {
      this.queue = []; // Initialize the queue as an empty array
    }
  
    // Method to add an element with a priority
    enqueue(element, priority) {
      let item = { element, priority }; // Create an object with element and its priority
      if (this.isEmpty()) {
        this.queue.push(item); // If the queue is empty, simply push the element
      } else {
        // If the queue is not empty, insert the element in the correct position
        let added = false;
        for (let i = 0; i < this.queue.length; i++) {
          // Find the correct position based on priority
          if (item.priority < this.queue[i].priority) {
            this.queue.splice(i, 0, item); // Insert the item at the correct index
            added = true;
            break;
          }
        }
        // If the element wasn't added yet, push it to the end
        if (!added) {
          this.queue.push(item);
        }
      }
    }
  
    // Method to remove and return the highest priority element (the front of the queue)
    dequeue() {
      if (this.isEmpty()) {
        return null; // Return null if the queue is empty
      }
      return this.queue.shift(); // Remove the first element (highest priority)
    }
  
    // Method to view the element with the highest priority without removing it
    peek() {
      if (this.isEmpty()) {
        return null; // Return null if the queue is empty
      }
      return this.queue[0].element; // Return the element with the highest priority
    }
  
    // Method to check if the queue is empty
    isEmpty() {
      return this.queue.length === 0; // Return true if the queue has no elements
    }
  
    // Method to get the size of the queue
    size() {
      return this.queue.length; // Return the number of elements in the queue
    }
  }
  
  // Example usage:
  let pq = new PriorityQueue();
  pq.enqueue('Task 1', 3); // Task 1 has priority 3
  pq.enqueue('Task 2', 1); // Task 2 has priority 1 (higher priority)
  pq.enqueue('Task 3', 2); // Task 3 has priority 2
  
  console.log(pq.dequeue()); // Removes and logs Task 2, the highest priority
  console.log(pq.peek());    // Logs Task 3, the next highest priority
  console.log(pq.size());    // Logs 1, because there's 1 element left in the queue