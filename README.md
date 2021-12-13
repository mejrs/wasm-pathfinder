To compile this project:

- Clone this project's parent repository: 
  ```text
  git clone --recurse-submodules https://github.com/mejrs/mejrs.github.io.git
   ```

- Install wasm-pack [here](https://rustwasm.github.io/wasm-pack/installer/#)

- Build the binary with
   ```text
   wasm-pack build --target web 
   ```