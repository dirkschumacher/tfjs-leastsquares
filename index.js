"use strict"

const tfc = require("@tensorflow/tfjs-core")
const backSolve = require("tfjs-backsolve")

const leastSquares = (X, y) => {
  return tfc.tidy(() => {
    let [q, r] = tfc.linalg.qr(X)
    const qty = tfc.matMul(q, y.reshape([X.shape[0], 1]), /* transpose_a */ true)
    const betaHat = backSolve(r, qty)
    return betaHat.reshape([X.shape[1], 1])
  })
}

module.exports = leastSquares