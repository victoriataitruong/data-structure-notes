// Creating nodes manually without a class definition
let node1 = { data: 10, next: null };
let node2 = { data: 20, next: null };
let node3 = { data: 30, next: null };

// Manually linking nodes
node1.next = node2;
node2.next = node3;

// Traversing the list
let current = node1;
while (current !== null) {
  console.log(current.data);
  current = current.next;
}