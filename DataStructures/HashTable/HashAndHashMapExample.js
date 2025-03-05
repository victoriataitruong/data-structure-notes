let hashMap = new Map();  // Creating a new Map

// Storing a key-value pair
hashMap.set("username", "codeMaster99");

// Retrieving the value using the key
console.log(hashMap.get("username")); // Output: codeMaster99

// Updating the value
hashMap.set("username", "devGuru88");
console.log(hashMap.get("username")); // Output: devGuru88

// Deleting a key-value pair
hashMap.delete("username");
console.log(hashMap.get("username")); // Output: undefined