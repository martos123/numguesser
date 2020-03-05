var mousePressed = false;
var lastX, lastY;
var ctx;
var model;
var arr = [];
var ex;

function InitThis() {
  document.getElementById("predict-number").disabled = true;
  document.getElementById("clear-area").disabled = true;

  document.getElementById("guess").innerHTML = "Guess: ";
  ctx = document.getElementById("myCanvas").getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  $("#myCanvas").mousedown(function(e) {
    mousePressed = true;
    Draw(
      e.pageX - $(this).offset().left,
      e.pageY - $(this).offset().top,
      false
    );
  });

  $("#myCanvas").mousemove(function(e) {
    if (mousePressed) {
      Draw(
        e.pageX - $(this).offset().left,
        e.pageY - $(this).offset().top,
        true
      );
    }
  });

  $("#myCanvas").mouseup(function(e) {
    mousePressed = false;
  });
  $("#myCanvas").mouseleave(function(e) {
    mousePressed = false;
  });
}

function Draw(x, y, isDown) {
  document.getElementById("predict-number").disabled = false;
  document.getElementById("clear-area").disabled = false;

  if (isDown) {
    ctx.beginPath();
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 15;
    ctx.lineJoin = "round";
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.closePath();
    ctx.stroke();
  }
  lastX = x;
  lastY = y;
}

function clearArea() {
  // Use the identity matrix while clearing the canvas
  document.getElementById("guess").innerHTML = "Guess: ";
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  document.getElementById("predict-number").disabled = true;
  document.getElementById("clear-area").disabled = true;
}

async function Predict() {
  model = await tf.loadLayersModel("model.json");
  example = document.getElementById("myCanvas");

  ex = tf.browser.fromPixels(example, 1);

  ex = tf.image.resizeBilinear(ex, [28, 28]);

  prediction = model.predict(ex.reshape([1, -1]));
  values = prediction.dataSync();
  arr = Array.from(values);
  maxim = argMax(arr);
  document.getElementById("guess").innerHTML = "Guess: " + maxim;
  document.getElementById("clear-area").disabled = false;
}
function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}
