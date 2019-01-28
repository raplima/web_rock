// inspired in https://hackernoon.com/classifying-images-using-tensorflow-js-keras-58431c4df04
// and files in https://github.com/ADLsourceCode/TensorflowJS
// to set up the server: open powershell in target directory -> http-server -c --cors

const ROCK_SAMPLES_CLASSES = {
    0: 'Basalt',
    1: 'Coquina',
    2: 'Garnet schist',
    3: 'Granite',
    4: 'Mica schist'
  };

$(document).ready()
{
  $('.progress-bar').hide();
}
$("#image-selector").change(function(){
    let reader = new FileReader();

    reader.onload = function(){
        let dataURL = reader.result;
        $("#selected-image").attr("src",dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});


$("#model-selector").change(function(){
    loadModel($("#model-selector").val());
    $('.progress-bar').show();
})

let model;
async function loadModel(name){
    //model=await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
	console.log('Model loaded')

	model=await tf.loadModel('http://localhost:8080/model/model.json')
    $('.progress-bar').hide();
}


$("#predict-button").click(async function(){
    let image= $('#selected-image').get(0);
    let tensor = preprocessImage(image,$("#model-selector").val());

    let prediction = await model.predict(tensor).data();
    let top5=Array.from(prediction)
                .map(function(p,i){
    return {
        probability: p,
        className: ROCK_SAMPLES_CLASSES[i]
    };
    }).sort(function(a,b){
        return b.probability-a.probability;
    }).slice(0,5);

	$("#prediction-list").empty();
	top5.forEach(function(p){
		$("#prediction-list").append(`<li>${p.className}:${p.probability.toFixed(2)}</li>`);
	});
	
	// update graph:
	// set up the graph 
	var datapoints = [];
	top5.forEach(function(p){
		datapoints.push({label: p.className, y: parseFloat(p.probability.toFixed(2))})
	});

	console.log(datapoints)
	var options = [{
	  animationEnabled: true,
	  theme: "light1",
	  title: {
		text: ""
	  },
	  axisX: {
		interval: 0.1
	  },
	  axisY2: {
		interlacedColor: "rgba(1,77,101,.2)",
		gridColor: "rgba(1,77,101,.1)",
		title: "Probability"
	  },
	  data: [{
		type: "bar",
		max: 1.0,
		axisYType: "secondary",
		color: "#006064",
		dataPoints: datapoints.reverse()
	  }]
	}];
	// this method take the class and options array 
	renderCharts("charts", options);
	
	
	// maybe use this graph:
	myX = []
	myY = []
	top5.forEach(function(p){
		myX.push(parseFloat(p.probability.toFixed(2)))
		myY.push(p.className)
	});
	
	var data = [{
		type: 'bar',
		x: myX.reverse(),
		y: myY.reverse(),
		orientation: 'h',
		marker: {
			color: "#006064",
			width: 1
		}
	}];
	Plotly.newPlot('plotChart', data, {}, {showSendToCloud:true});
	
});

function renderCharts(className, options) {
  var charts = [];
  var chartClassElements = document.getElementsByClassName(className);
  for (var i = 0; i < chartClassElements.length; i++) {
    charts.push(new CanvasJS.Chart(chartClassElements[i], options[i]));
    charts[i].render();
  }
}


function preprocessImage(image,modelName)
{
    let tensor=tf.fromPixels(image)
    .resizeBilinear([224,224])
    .toFloat();
          
    if(modelName==undefined)
    {
        return tensor.expandDims();
    }
    else if(modelName=="mobilenet")
    {
        let offset=tf.scalar( 1. / 255);
        return tensor.mul(offset)
                    .expandDims();
    }
    else
    {
        throw new Error("UnKnown Model error");
    }
}