function RNNViewer(settings) {
  Object.assign(this, RNNViewer.defaults, settings);

  this.net = settings.net;
  this.boundingGrid = null;
  this.values = [];
  this.grids = [];
  this.matrices = [];
  this.controls = null;
  this.scene = null;
  this.camera = null;
  this.light = null;
  this.renderer = null;
  this.stats = null;

  this.init();

  if (this.net) {
    var model = this.net.model;
    var addMatrix = this.addMatrix.bind(this);

    addMatrix(model.input);

    model.hiddenLayers.forEach(function(hiddenLayer) {
      for (var p in hiddenLayer) {
        if (!hiddenLayer.hasOwnProperty(p)) continue;
        addMatrix(hiddenLayer[p]);
      }
    });

    addMatrix(model.outputConnector);
    addMatrix(model.output);
  }

  this.animate();
}

RNNViewer.defaults = {
  net: null,
  container: null,
  height: window.innerHeight,
  width: window.innerWidth,
  depth: 100,
  hotColor: new THREE.Color(0xff55f9),
  coldColor: new THREE.Color(0x050638),
  squareWidth: 10,
  squareHeight: 10,
  devicePixelRatio: window.devicePixelRatio,
  includeStats: false
};

RNNViewer.prototype = {
  init: function() {
    //Set up camera
    var vFOVRadians = 2 * Math.atan(this.height / (2 * 1500)),
      fov = vFOVRadians * 180 / Math.PI,
      startPosition = this.startPosition = new THREE.Vector3(0, 0, 3000);

    var camera = this.camera = new THREE.PerspectiveCamera(fov, this.width / this.height, 1, 30000);
    camera.position.set(startPosition.x, startPosition.y, startPosition.z);

    var controls = this.controls = new THREE.OrbitControls(camera);
    controls.damping = 0.2;
    controls.addEventListener('change', this.render.bind(this));

    //Create scenes for webGL
    var scene = this.scene = new THREE.Scene();
    //Add a light source & create Canvas
    var light = this.light = new THREE.DirectionalLight( 0xffffff );
    light.position.set(0, 0, 1);
    scene.add(light);

    //set up webGL renderer
    var renderer = this.renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(this.devicePixelRatio);
    renderer.setSize(this.width, this.height);
    this.container.appendChild(renderer.domElement);

    //stats
    if (this.includeStats) {
      var stats = this.stats = new Stats();
      stats.domElement.style.position = 'absolute';
      stats.domElement.style.bottom = '10px';
      stats.domElement.style.left = '10px';
      this.container.appendChild(stats.domElement);
    }

    var boundingGrid = this.boundingGrid = new THREE.Object3D();
    scene.add(boundingGrid);
    return this;
  },
  update: function() {
    var hotColor = this.settings.hotColor;
    var coldColor = this.settings.coldColor;
    return this;
  },
  render: function() {
    var depth = this.depth;
    this.grids.forEach(function(grid, i, grids) {
      grid.position.z = (grids.length - i) * depth;
    });

    this.camera.lookAt(this.scene.position);
    this.renderer.render(this.scene, this.camera);
    if (this.stats) this.stats.update();
    return this;
  },
  animate: function() {
    this.controls.update();
    window.requestAnimationFrame(this.animate.bind(this));
    return this;
  },
  addMatrix: function (matrix) {
    var grid = new THREE.Object3D(),
      depth = this.depth,
      rows = matrix.rows,
      columns = matrix.columns,
      xPixel = -(this.squareWidth * columns)/ 2,
      yPixel = -(this.squareHeight * rows) / 2,
      lowValue = 0,
      highValue = 0,
      index = 0;

    //height
    for (var row = 1; row <= rows; row++) {
      xPixel = -(this.squareWidth * columns) / 2;
      for (var column = 1; column <= columns; column++) {
        var color = this.coldColor.clone();
        var material = new THREE.MeshBasicMaterial({
          color: color,
          side: THREE.DoubleSide,
          vertexColors: THREE.FaceColors
        });
        var square = new THREE.Geometry();
        square.vertices.push(new THREE.Vector3(xPixel                   , yPixel                     , 0));
        square.vertices.push(new THREE.Vector3(xPixel                   , yPixel + this.squareHeight , 0));
        square.vertices.push(new THREE.Vector3(xPixel + this.squareWidth, yPixel + this.squareHeight , 0));
        square.vertices.push(new THREE.Vector3(xPixel + this.squareWidth, yPixel                     , 0));

        square.faces.push(new THREE.Face3(0, 1, 2));
        square.faces.push(new THREE.Face3(0, 3, 2));
        var mesh = new THREE.Mesh(square, material);
        grid.add(mesh);

        this.values.push({
          color: color,
          row: row - 1,
          column: column - 1,
          matrixIndex: this.grids.length,
          square: square,
          mesh: mesh,
          frontFace: mesh.geometry.faces[0],
          rearFace: mesh.geometry.faces[1],
          index: index,
          matrix: matrix,
          get value() {
            var value = this.matrix.weights[this.index];
            if (value > highValue) {
              highValue = value;
            }
            if (value < lowValue) {
              lowValue = value;
            }
            return value || 0;
          },
          get percentValue() {
            var value = this.value;
            var normalizedHigh = highValue - lowValue;
            var normalizedValue = value - lowValue;
            return (normalizedHigh - normalizedValue) / normalizedHigh;
          }
        });

        xPixel += this.squareWidth;
        index++;
      }
      yPixel += this.squareHeight;
    }

    this.grids.push(grid);
    this.matrices.push(matrix);
    this.boundingGrid.add(grid);

    return this;
  },
  viewTop: function() {
    this.controls.reset();

    var vFOVRadians = 2 * Math.atan(this.height / ( 2 * 35000 )),
      fov = vFOVRadians * 180 / Math.PI;

    this.camera.fov = fov;
    this.controls.rotateUp(90 * Math.PI / 180);
    this.camera.position.z = this.startPosition.z * 23;
    this.camera.position.y = this.startPosition.z * 55;
    this.camera.far = 1000000;
    this.camera.updateProjectionMatrix();
    return this.render();
  },
  viewSide: function() {
    this.controls.reset();

    var vFOVRadians = 2 * Math.atan(this.height / ( 2 * 35000 )),
      fov = vFOVRadians * 180 / Math.PI;

    this.camera.fov = fov;
    this.camera.position.z = this.startPosition.z * 58;
    this.camera.far = 1000000;
    this.camera.updateProjectionMatrix();
    return this.render();
  },
  viewDefault: function() {
    this.controls.reset();

    this.camera.fov = 30;
    this.camera.updateProjectionMatrix();
    return this.render();
  },
  setSize: function(width, height) {
    this.width = width;
    this.height = height;
    this.renderer.setSize(this.width, this.height);
    return this.render();
  },
  setValue: function(v) {
    var v = Math.random() * 2,
      r = (coldColor.r + hotColor.r) / v,
      g = (coldColor.g + hotColor.g) / v,
      b = (coldColor.b + hotColor.b) / v;

    value.frontFace.color.setRGB(
      r,
      g,
      b
    );
    value.rearFace.color.setRGB(
      r,
      g,
      b
    );
    value.square.colorsNeedUpdate = true;
    //value.mesh.geometry.elementsNeedUpdate = true;
    value.mesh.geometry.colorsNeedUpdate = true;
  }
};