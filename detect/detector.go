package main

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"sort"
	"time"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"gocv.io/x/gocv"
)

var (
	graph          *tf.Graph
	labels         []string
	detectedResult = ""
	isLabio        = false
)

func main() {
	faceImgChan := make(chan string, 100)
	go startImageProcessor(faceImgChan)
	go func() {
		for file := range faceImgChan {
			if err := os.Remove(file); err != nil {
				fmt.Println(err)
			}
		}
	}()

	go startCameraFeed(faceImgChan)
	select {}
}

func startImageProcessor(imgChan chan string) {
	//
	if err := loadModel(); err != nil {
		log.Fatal(err)
		return
	}

	labels := make(map[string]float32)
	for imageFile := range imgChan {
		//
		labelResult, err := getProbabilities(imageFile)
		if err != nil {
			log.Printf("startImageProcessor(): %s", err)
			continue
		}
		fmt.Printf("----TensorFlow:\n")
		if labelResult[0].Label == "labio" && labelResult[0].Probability > 0.5 {
			isLabio = true
			playSiren()
		}
		for _, lr := range labelResult {
			labels[lr.Label] = lr.Probability
			fmt.Printf("%s = %2.4f\n", lr.Label, lr.Probability)
		}
		detectedResult = fmt.Sprintf("Labio: %1.3f; Normal: %1.3f;", labels["labio"], labels["normal"])

		// delete image
		if err := os.Remove(imageFile); err != nil {
			fmt.Println(err)
		}
	}
}

func getProbabilities(imageFile string) ([]LabelResult, error) {

	b, err := ioutil.ReadFile(imageFile)
	if err != nil {
		return nil, err
	}
	imageBuffer := bytes.NewBuffer(b)

	tensor, err := makeTensorFromImage(imageBuffer, "jpg")
	if err != nil {
		return nil, fmt.Errorf("Invalid image")
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, fmt.Errorf("Error creating session: %s", err)
	}
	defer session.Close()

	/*
		fmt.Println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		fmt.Println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		for _, o := range graph.Operations() {
			fmt.Println(o.Name())
		}
		fmt.Println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		fmt.Println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
	*/

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{graph.Operation("input").Output(0): tensor},
		[]tf.Output{graph.Operation("final_result").Output(0)},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("Could not run inference: %s", err)
	}
	// The output[0].Value() tensor now contains probabilities of each label
	return findBestLabels(output[0].Value().([][]float32)[0]), nil
}

func loadModel() error {
	// Load inception model
	model, err := ioutil.ReadFile("retrained_graph.pb")
	if err != nil {
		return err
	}
	graph = tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return err
	}
	// Load labels
	labelsFile, err := os.Open("retrained_labels.txt")
	if err != nil {
		return err
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)
	// Labels are separated by newlines
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

func makeTensorFromImage(imageBuffer *bytes.Buffer, imageFormat string) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(imageBuffer.String())
	if err != nil {
		return nil, err
	}
	graph, input, output, err := makeTransformImageGraph(imageFormat)
	if err != nil {
		return nil, err
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func makeTransformImageGraph(imageFormat string) (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		H, W = 224, 224
		//Mean  = float32(117)
		Mean  = float32(128)
		Scale = float32(1)
	)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	// Decode PNG or JPEG
	var decode tf.Output
	if imageFormat == "png" {
		decode = op.DecodePng(s, input, op.DecodePngChannels(3))
	} else {
		decode = op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
	}
	// Div and Sub perform (value-Mean)/Scale for each pixel
	output = op.Div(
		s,
		op.Sub(
			s,
			// Resize to 224x224 with bilinear interpolation
			op.ResizeBilinear(
				s,
				// Create a batch containing a single image
				op.ExpandDims(
					s,
					// Use decoded pixel values
					op.Cast(s, decode, tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0)),
				),
				op.Const(s.SubScope("size"), []int32{H, W}),
			),
			op.Const(s.SubScope("mean"), Mean),
		),
		op.Const(s.SubScope("scale"), Scale),
	)
	graph, err = s.Finalize()
	return graph, input, output, err
}

// LabelResult result of operation
type LabelResult struct {
	Label       string  `json:"label"`
	Probability float32 `json:"probability"`
}

// ByProbability sorter
type ByProbability []LabelResult

func (a ByProbability) Len() int           { return len(a) }
func (a ByProbability) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByProbability) Less(i, j int) bool { return a[i].Probability > a[j].Probability }

func findBestLabels(probabilities []float32) []LabelResult {
	// Make a list of label/probability pairs
	var resultLabels []LabelResult
	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}
		resultLabels = append(resultLabels, LabelResult{Label: labels[i], Probability: p})
	}
	// Sort by probability
	sort.Sort(ByProbability(resultLabels))
	// Return top 2 labels
	return resultLabels[:2]
}

var (
	id       = 1
	pre      = rand.New(rand.NewSource(time.Now().UnixNano())).Int()
	deviceID = 0
	xmlFile  = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
)

func startCameraFeed(faceImgChan chan string) {
	// open webcam
	webcam, err := gocv.VideoCaptureDevice(int(deviceID))
	if err != nil {
		fmt.Println(err)
		return
	}
	defer func() { _ = webcam.Close() }()

	detector := newFaceDetector(xmlFile)
	defer detector.close()

	showCamera(webcam, detector, faceImgChan)
}
func showCamera(webcam *gocv.VideoCapture, detector *faceDetector, faceImgChan chan string) {
	// open display window
	window := gocv.NewWindow("Face Detect")
	window.ResizeWindow(512, 512)
	defer func() { _ = window.Close() }()

	// prepare image matrix
	img := gocv.NewMat()
	defer func() { _ = img.Close() }()

	// color for the rect when faces detected
	blue := color.RGBA{0, 0, 255, 0}

	/*
		imgChan := make(chan string, 100)
		scoreChan := make(chan map[string]float32, 100)
		go func() {
			for scoreMap := range scoreChan {
				fmt.Println("=============================")
				for k, v := range scoreMap {
					fmt.Printf("%s->%2.4f\n", k, v)
				}
			}
		}()
		go startImageLabeler(imgChan, scoreChan)
	*/
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))

	//
	for {
		if ok := webcam.Read(&img); !ok {
			panic("cannot read device\n")
		}
		if img.Empty() {
			continue
		}

		// detect faces
		detector.detect(img, func(rects []image.Rectangle) {
			//fmt.Printf("found %d faces\n", len(rects))

			// draw a rectangle around each face on the original image,
			// along with text identifying as "Human"
			for _, r := range rects {

				//
				//
				faceImage := img.Region(r)
				defer func() { _ = faceImage.Close() }()
				faceImageName := path.Join(
					"./images",
					fmt.Sprintf("test-%d.jpg", rnd.Int()),
				)
				gocv.IMWrite(faceImageName, faceImage)
				faceImgChan <- faceImageName
				//

				txt := "Human"
				if detectedResult != "" {
					txt = detectedResult
				}
				gocv.Rectangle(&img, r, blue, 3)
				size := gocv.GetTextSize(txt, gocv.FontHersheyPlain, 1.2, 2)
				pt := image.Pt(r.Min.X+(r.Min.X/2)-(size.X/2), r.Min.Y-2)
				gocv.PutText(&img, txt, pt, gocv.FontHersheyPlain, 1.2, blue, 2)
			}

			// show the image in the window, and wait 1 millisecond
			window.IMShow(img)

		})
		if window.WaitKey(1) >= 0 {
			break
		}
		//time.Sleep(time.Millisecond * 2000) // sleep 1 second
	}
}

type faceDetector struct {
	inFlight   bool
	closed     bool
	classifier gocv.CascadeClassifier
}

func newFaceDetector(xmlFile string) *faceDetector {
	// load classifier to recognize faces
	classifier := gocv.NewCascadeClassifier()

	if !classifier.Load(xmlFile) {
		fmt.Printf("Error reading cascade file: %v\n", xmlFile)
		panic("error")
	}
	return &faceDetector{classifier: classifier, inFlight: false}
}

func (fd *faceDetector) detect(img gocv.Mat, f func([]image.Rectangle)) {
	if fd.inFlight || fd.closed {
		return
	}

	fd.inFlight = true
	go func() {
		imgCopy := gocv.NewMat()
		defer func() { _ = imgCopy.Close() }()
		img.CopyTo(&imgCopy)
		rects := fd.classifier.DetectMultiScale(imgCopy)
		//fmt.Printf("Dtected %d faces", len(rects))
		if !fd.closed {
			f(rects)
		}
		fd.inFlight = false
	}()

}

func (fd *faceDetector) close() {
	defer func() { _ = fd.classifier.Close() }()
}

func playSiren() {
	var playCmd string
	var err error
	playCmd, err = exec.LookPath("play")
	if err != nil {
		panic(err)
	}
	cmd := exec.Command(playCmd, "siren.wav")
	err = cmd.Run()
	if err != nil {
		log.Println(err)
	}
}
