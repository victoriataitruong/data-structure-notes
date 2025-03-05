let hashTable = {};  // Creating an empty object

// Storing a key-value pair
hashTable["username"] = "codeMaster99";

// Retrieving the value using the key
console.log(hashTable["username"]); // Output: codeMaster99

// Updating the value
hashTable["username"] = "devGuru88";
console.log(hashTable["username"]); // Output: devGuru88

// Deleting a key-value pair
delete hashTable["username"];
console.log(hashTable["username"]); // Output: undefined
  