"use strict"

const test = require("tape")
const tfc = require("@tensorflow/tfjs-core")
const fit = require(".")
const mtcars = require("mtcars")
const round = require("lodash.round")

test("fit mtcars", (t) => {
  const mpg = mtcars.map((x) => x.mpg)
  const n = mpg.length
  const m = 2
  const hp = mtcars.map((x) => x.hp)
  const cyl = mtcars.map((x) => x.cyl)
  const response = tfc.tensor1d(mpg)
  
  const designMatrix = tfc.tensor2d(
    [hp, cyl],
    [m, n]
  ).transpose()

  // fit the model
  const coefficents = fit(designMatrix, response)

  // computed with R 3.4.2
  // fited coefficents
  const expectedHp = -0.107465705415024
  const expectedCyl = 5.403644695759401

  const arrayEqual = (a, b) => {
    for(let i = 0; i < n; i++) {
      t.equal(a[i], b[i])
    }
  }
  const rm = (x) => round(x, 4)
  arrayEqual(coefficents.arraySync().map(rm), [expectedHp, expectedCyl].map(rm))

  t.end()
})