class SimpleMaxHeap {
  constructor() {
    this.heap = []; // Initialize the heap as an empty array
  }

  // Insert a patient into the heap
  insert(patient) {
    this.heap.push(patient); // Add the patient to the end of the heap
    this._heapifyUp(); // Reorganize the heap to maintain the heap property
  }

  // Move the last inserted patient up to its correct position
  _heapifyUp() {
    let index = this.heap.length - 1; // Start from the last patient
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2); // Find the parent index
      // If the current patient's urgency is greater than the parent's, swap them
      if (this.heap[index].urgency > this.heap[parentIndex].urgency) {
        [this.heap[index], this.heap[parentIndex]] = [
          this.heap[parentIndex],
          this.heap[index],
        ];
        index = parentIndex; // Move up to the parent's position
      } else {
        break; // No need to move further up, the heap property is satisfied
      }
    }
  }

  // Extract the most urgent patient (the root of the heap)
  extractMax() {
    if (this.heap.length === 0) return null; // If the heap is empty, return null

    // Swap the root with the last patient
    const mostUrgent = this.heap[0];
    this.heap[0] = this.heap[this.heap.length - 1];
    this.heap.pop(); // Remove the last patient

    this._heapifyDown(); // Reorganize the heap to maintain the heap property
    return mostUrgent; // Return the most urgent patient
  }

  // Move the root patient down to its correct position
  _heapifyDown() {
    let index = 0; // Start from the root
    while (index * 2 + 1 < this.heap.length) {
      let leftChildIndex = index * 2 + 1; // Left child index
      let rightChildIndex = index * 2 + 2; // Right child index
      let largest = index;

      // Find the largest child (if any)
      if (
        leftChildIndex < this.heap.length &&
        this.heap[leftChildIndex].urgency > this.heap[largest].urgency
      ) {
        largest = leftChildIndex;
      }
      if (
        rightChildIndex < this.heap.length &&
        this.heap[rightChildIndex].urgency > this.heap[largest].urgency
      ) {
        largest = rightChildIndex;
      }

      // If the largest child has a higher urgency, swap with the current patient
      if (largest !== index) {
        [this.heap[index], this.heap[largest]] = [
          this.heap[largest],
          this.heap[index],
        ];
        index = largest; // Move down to the child's position
      } else {
        break; // No need to move further down
      }
    }
  }
}

// Create a new max heap for patients
const heap = new SimpleMaxHeap();

// Insert some patients with their urgency levels
heap.insert({ name: "Alice", urgency: 10 }); // Critical
heap.insert({ name: "Bob", urgency: 3 }); // Mild Fever
heap.insert({ name: "Charlie", urgency: 7 }); // Broken Arm

// Extract and treat patients based on their urgency
console.log(heap.extractMax()); // Alice (Critical) - most urgent
console.log(heap.extractMax()); // Charlie (Broken Arm) - second most urgent
console.log(heap.extractMax()); // Bob (Mild Fever) - least urgent
