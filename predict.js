// inspired in https://hackernoon.com/classifying-images-using-tensorflow-js-keras-58431c4df04
// and files in https://github.com/ADLsourceCode/TensorflowJS
// to set up the server: open powershell in target directory -> http-server -c --cors

const ROCK_SAMPLES_CLASSES = {
	0: 'Barite rose', 
    1: 'Basalt',
    2: 'Coquina',
    3: 'Garnet schist',
    4: 'Granite',
    5: 'Mica schist'
  };

$(document).ready()
{
  $('.progress-bar').hide();
}
$("#image-selector").change(async function(){
    let reader = new FileReader();

    reader.onload = function(){
        let dataURL = reader.result;
        $("#selected-image").attr("src",dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
	
	// make sure image is loaded, then predict and plot
	reader.onloadend = async() => {
		console.log('Image loaded')
		
		// predict
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
		
		// update the list:
		$("#prediction-list").empty();
		top5.forEach(function(p){
			$("#prediction-list").append(`<li>${p.className}:${p.probability.toFixed(2)}</li>`);
		});
		
		// update graph:
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
		Plotly.newPlot('plotChart', data, {}, {displayModeBar: false});
	
	}
	
});


$("#model-selector").change(function(){
    loadModel($("#model-selector").val());
    $('.progress-bar').show();
})

let model;
async function loadModel(name){
	
	model=await tf.loadModel(`http://localhost:8080/model/${name}/model.json`)
	//model=await tf.loadModel(`model/${name}/model.json`)
    $('.progress-bar').hide();
	console.log('Model loaded')
	
}

function preprocessImage(image,modelName)
{
              
    if(modelName==undefined)
    {	
		let tensor=tf.fromPixels(image)
			.resizeBilinear([224,224])
			.toFloat();
        return tensor.expandDims();
    }
    else if(modelName=="MobileNetV2")
    {
		let tensor=tf.fromPixels(image)
			.resizeBilinear([224,224])
			.toFloat();
        let offset=tf.scalar( 1. / 255);
        return tensor.mul(offset)
                    .expandDims();
    }
	else if(modelName=="VGG19")
    {	
		let tensor=tf.fromPixels(image)
			.resizeBilinear([224,224])
			.toFloat();
        let offset=tf.scalar( 1. / 255);
        return tensor.mul(offset)
                    .expandDims();
    }
	else if(modelName=="InceptionV3")
    {   
		let tensor=tf.fromPixels(image)
			.resizeBilinear([299,299])
			.toFloat();

        let offset=tf.scalar( 1. / 255);
        return tensor.mul(offset)
                    .expandDims();
    }
    else
    {
        throw new Error("UnKnown Model error");
    }
}