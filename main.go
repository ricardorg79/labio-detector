package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"time"

	"gocv.io/x/gocv"
)

var (
	id  = 1
	pre = rand.New(rand.NewSource(time.Now().UnixNano())).Int()
	//haarXMLFilePtr = flag.String("xml", "", "Haar Cascade XML File")
	outputDir      = flag.String("output-dir", "", "Directory to Store captured images")
	deviceID       = 0
	xmlClassifiers stringArgs
	classifiers    []gocv.CascadeClassifier
)

type stringArgs []string

func (a *stringArgs) String() string {
	return "My String Representation"
}
func (a *stringArgs) Set(val string) error {
	*a = append(*a, val)
	return nil
}

func main() {
	flag.Var(&xmlClassifiers, "xml", "Haar Cascade XML File")
	flag.Parse()

	/*
		if len(os.Args) < 3 {
			fmt.Println("How to run:\n\tfacedetect [camera ID] [classifier XML file]")
			return
		}

		// parse args
		deviceID, _ := strconv.Atoi(os.Args[1])
		xmlFile := os.Args[2]
	*/
	//xmlFile := "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
	if len(xmlClassifiers) < 1 {
		fmt.Printf("-------------------------\nXml File required\n-------------------------\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	for _, xmlFile := range xmlClassifiers {
		classifier := gocv.NewCascadeClassifier()
		if !classifier.Load(xmlFile) {
			fmt.Printf("Error reading cascade file: %v\n", xmlFile)
			panic("error")
		}
		classifiers = append(classifiers, classifier)
	}

	// open webcam
	webcam, err := gocv.VideoCaptureDevice(int(deviceID))
	if err != nil {
		fmt.Println(err)
		return
	}
	defer func() { _ = webcam.Close() }()

	detector := newFaceDetector(xmlFile)
	defer detector.close()
	showCamera(webcam, detector)
	select {}
}

func showCamera(webcam *gocv.VideoCapture, detector *faceDetector) {
	// open display window
	window := gocv.NewWindow("Face Detect")
	window.ResizeWindow(800, 800)
	defer func() { _ = window.Close() }()

	// prepare image matrix
	img := gocv.NewMat()
	defer func() { _ = img.Close() }()

	// color for the rect when faces detected
	blue := color.RGBA{0, 0, 255, 0}

	//fmt.Printf("start reading camera device: %v\n", deviceID)
	for {
		if ok := webcam.Read(&img); !ok {
			panic("cannot read device\n")
		}
		if img.Empty() {
			continue
		}
		//gocv.Flip(img, &img, 0)

		// detect faces
		detector.detect(img, func(rects []image.Rectangle) {
			fmt.Printf("found %d faces\n", len(rects))

			// draw a rectangle around each face on the original image,
			// along with text identifying as "Human"
			for _, r := range rects {

				if *outputDir != "" {
					//
					faceImage := img.Region(r)
					defer func() { _ = faceImage.Close() }()
					faceImageName := fmt.Sprintf("%d-%05d.jpg", pre, id)
					gocv.IMWrite(path.Join(*outputDir, faceImageName), faceImage)
					id++
				}

				//
				txt := "Detected"
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

func newFaceDetector() *faceDetector {
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
		fmt.Printf("Dtected %d faces", len(rects))
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
		panic(err)
	}

	/*
		f, err := os.Open("siren.wav")
		if err != nil {
			fmt.Printf("Error opening siren.wav: %s", err)
		}
		defer f.Close()
		s, format, _ := wav.Decode(f)
		if err := speaker.Init(format.SampleRate, format.SampleRate.N(time.Second/10)); err != nil {
			fmt.Printf("Error initializing speaker: %s", err)
		}
		//speaker.Play(s)
		speaker.Play(beep.Seq(s, beep.Callback(func() {
			//
		})))
	*/

}
