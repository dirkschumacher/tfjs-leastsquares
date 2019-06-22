"use strict"

const tfc = require("@tensorflow/tfjs-core")
const backSolve = require("tfjs-backsolve")

const leastSquares = async (X, y) => {
  let [q, r] = tfc.linalg.qr(X)
  const qty = tfc.matMul(q, y.reshape([X.shape[0], 1]), /* transpose_a */ true)
  const betaHat = await backSolve(r, qty)
  q.dispose()
  r.dispose()
  qty.dispose()
  return betaHat
}

module.exports = leastSquares