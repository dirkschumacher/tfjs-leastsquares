# tfjs-leastsquares
Solve linear least-squares problems in tensorflow.js

```js
const tfc = require("@tensorflow/tfjs-core")
const fit = require("tfjs-leastsquares")
const mtcars = require("mtcars")

// build the input data
const n = mpg.length
const m = 2
const mpg = mtcars.map((x) => x.mpg)
const hp = mtcars.map((x) => x.hp)
const cyl = mtcars.map((x) => x.cyl)
const response = tfc.tensor1d(mpg)  
const designMatrix = tfc.tensor2d(
  [hp, cyl],
  [m, n]
).transpose()

// fit the model
const coefficents = fit(designMatrix, response)
// tensor2d: [[-0.107465705415024], [5.403644695759401]]
```
