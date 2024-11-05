import { useEffect, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import * as onnx from 'onnxjs';
import * as tf from '@tensorflow/tfjs';

function App() {
const [naam, setNaam] = useState([])
const [num,setNum]=useState(0)

async function run(num) {  
  // var a=[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
  // // console.log(a)
  //   // Create input tensor
  // const inputTensor = new onnx.Tensor(new Float32Array(a.flat()), 'float32',[1,28]);
  // // console.log(inputTensor)
  // // // Run inference
  // const output = await session.run([inputTensor]);
  // var typedArray=new Float32Array(output.get('6').data)
  // var tf_tensor=tf.tensor(typedArray, [1, 28])
  // // console.log(new Float32Array(output.get('6').data))
  // // console.log(await tf_tensor.data())
  // var x=await tf.oneHot(tf.tensor1d([0]).toInt(),28)
  // var dataarray=x.dataSync()
  // var onnxtensor = new onnx.Tensor(dataarray, 'int32',[1,28])
  const stoi={' ': 1,'a': 2,'b': 3,'c': 4,'d': 5,'e': 6,'f': 7,'g': 8,'h': 9,'i': 10,'j': 11,'k': 12,'l': 13,'m': 14,'n': 15,
    'o': 16,'p': 17,'q': 18,'r': 19,'s': 20,'t': 21,'u': 22,'v': 23,'w': 24,'x': 25,'y': 26,'z': 27,'.': 0}
  const itos={1: ' ',2: 'a',3: 'b',4: 'c',5: 'd',6: 'e',7: 'f',8: 'g',9: 'h',10: 'i',11: 'j',12: 'k',13: 'l',14: 'm',15: 'n',
    16: 'o',17: 'p',18: 'q',19: 'r',20: 's',21: 't',22: 'u',23: 'v',24: 'w',25: 'x',26: 'y',27: 'z',0: '.'}
  // console.log(onnxtensor.data)
  var out=[]
  const session = new onnx.InferenceSession();
  await session.loadModel('test_model.onnx');
  for(var i=0;i<num;i++){
    var ix=0
    var xin=0
    var str=''
    while(true){
      
      var xenc=await tf.oneHot(tf.tensor1d([xin]).toInt(),28)
      var dataarray=xenc.dataSync()
      dataarray=new Float32Array(dataarray)
      var onnxtensor = new onnx.Tensor(dataarray, 'float32',[1,28])
      const output = await session.run([onnxtensor]);
      var typedArray=new Float32Array(output.get('6').data)
      var arr=await tf.tensor(typedArray,[1,28])
      var pick=tf.multinomial(arr,1,0,true)
      var o=await pick.data()
      // console.log(typedArray)
      // console.log(itos[typedArray.indexOf(Math.max(...typedArray))])
      str=str+itos[o[0]]
      xin=o[0]
      // console.log(itos[o])
      if(xin==0){
        break
      }
    }
    // console.log(str)
    out.push(str)
  }
  // console.log(out)
  return out
}
const chan=(e)=>{
  var c=e.target.value
  if(c>10){
    c=10
  }
  setNum(c)
}

const clicked=async ()=>{
  var names=await run(num)
  console.log(names)
  setNaam(names)
  console.log(naam)
}
useEffect(()=>{
  // run()
},[])

if(naam){
  return (
    <div>
      <h1>Not so accurate Indian Name generator 😁</h1>
      <ul>
        {naam.map((str,i)=>(
          <h4 >{str}</h4>
        ))
        }
      </ul>
      <input style={{paddingTop:10,paddingBottom:10, marginRight:10,border: '1px solid black',borderRadius: 5,}} placeholder='Ex:10' onChange={chan}/>
      <button onClick={clicked}>Generate</button>
      <p><i>psst enter number of names to generate  can`t generate more than 10 in one go</i></p>
    </div>
  )
}
else{
  return(
    <h1>Loading ...</h1>
  )
}
  
}

export default App
