# -*- coding: utf-8 -*-
"""
Shared qtpynodeeditor integration for xrtGlow and xrtQook.
"""

from collections import defaultdict

from ...commons import qt

try:
    from qtpynodeeditor import (
        FlowScene, FlowView, NodeData, NodeDataModel, NodeDataType,
        PortType, Node, NodeGraphicsObject, StyleCollection,
        ConnectionGraphicsObject, NodeConnectionInteraction)
except ImportError:  # pragma: no cover - optional dependency
    FlowScene = FlowView = NodeData = NodeDataModel = NodeDataType = None
    PortType = Node = NodeGraphicsObject = StyleCollection = None
    ConnectionGraphicsObject = NodeConnectionInteraction = None

HAS_QTPYNODEEDITOR = FlowScene is not None


if StyleCollection is not None:
    FLOW_SCENE_STYLE = StyleCollection.from_json({
        "FlowViewStyle": {
            "BackgroundColor": [30, 37, 43],
            "FineGridColor": [39, 49, 58],
            "CoarseGridColor": [17, 23, 28]
        },
        "NodeStyle": {
            "NormalBoundaryColor": [180, 190, 198],
            "SelectedBoundaryColor": [255, 209, 102],
            "GradientColor0": [102, 113, 122],
            "GradientColor1": [82, 92, 100],
            "GradientColor2": [62, 71, 79],
            "GradientColor3": [47, 55, 61],
            "ShadowColor": [10, 14, 18, 180],
            "FontColor": [243, 246, 248],
            "FontColorFaded": [185, 193, 199],
            "ConnectionPointColor": [176, 189, 198],
            "FilledConnectionPointColor": [143, 183, 201],
            "ErrorColor": [214, 95, 95],
            "WarningColor": [230, 162, 60],
            "PenWidth": 1.2,
            "HoveredPenWidth": 1.8,
            "ConnectionPointDiameter": 8.0,
            "Opacity": 0.96
        },
        "ConnectionStyle": {
            "ConstructionColor": [120, 138, 148],
            "NormalColor": [143, 183, 201],
            "SelectedColor": [255, 209, 102],
            "SelectedHaloColor": [255, 209, 102],
            "HoveredColor": [198, 227, 238],
            "LineWidth": 3.0,
            "ConstructionLineWidth": 2.0,
            "PointDiameter": 10.0,
            "UseDataDefinedColors": False
        }
    })

    FLOW_NODE_STYLES = {
        'source': StyleCollection.from_json({
            "FlowViewStyle": {
                "BackgroundColor": [30, 37, 43],
                "FineGridColor": [39, 49, 58],
                "CoarseGridColor": [17, 23, 28]
            },
            "NodeStyle": {
                "NormalBoundaryColor": [97, 214, 180],
                "SelectedBoundaryColor": [255, 209, 102],
                "GradientColor0": [95, 216, 181],
                "GradientColor1": [31, 161, 135],
                "GradientColor2": [24, 131, 110],
                "GradientColor3": [18, 97, 82],
                "ShadowColor": [10, 14, 18, 180],
                "FontColor": [245, 250, 249],
                "FontColorFaded": [190, 214, 208],
                "ConnectionPointColor": [97, 214, 180],
                "FilledConnectionPointColor": [143, 183, 201],
                "ErrorColor": [214, 95, 95],
                "WarningColor": [230, 162, 60],
                "PenWidth": 1.3,
                "HoveredPenWidth": 1.9,
                "ConnectionPointDiameter": 8.0,
                "Opacity": 0.98
            },
            "ConnectionStyle": {
                "ConstructionColor": [120, 138, 148],
                "NormalColor": [143, 183, 201],
                "SelectedColor": [255, 209, 102],
                "SelectedHaloColor": [255, 209, 102],
                "HoveredColor": [198, 227, 238],
                "LineWidth": 3.0,
                "ConstructionLineWidth": 2.0,
                "PointDiameter": 10.0,
                "UseDataDefinedColors": False
            }
        }),
        'oe': StyleCollection.from_json({
            "FlowViewStyle": {
                "BackgroundColor": [30, 37, 43],
                "FineGridColor": [39, 49, 58],
                "CoarseGridColor": [17, 23, 28]
            },
            "NodeStyle": {
                "NormalBoundaryColor": [117, 163, 214],
                "SelectedBoundaryColor": [255, 209, 102],
                "GradientColor0": [116, 157, 210],
                "GradientColor1": [76, 120, 168],
                "GradientColor2": [58, 94, 135],
                "GradientColor3": [42, 68, 98],
                "ShadowColor": [10, 14, 18, 180],
                "FontColor": [245, 247, 250],
                "FontColorFaded": [192, 201, 214],
                "ConnectionPointColor": [117, 163, 214],
                "FilledConnectionPointColor": [143, 183, 201],
                "ErrorColor": [214, 95, 95],
                "WarningColor": [230, 162, 60],
                "PenWidth": 1.3,
                "HoveredPenWidth": 1.9,
                "ConnectionPointDiameter": 8.0,
                "Opacity": 0.98
            },
            "ConnectionStyle": {
                "ConstructionColor": [120, 138, 148],
                "NormalColor": [143, 183, 201],
                "SelectedColor": [255, 209, 102],
                "SelectedHaloColor": [255, 209, 102],
                "HoveredColor": [198, 227, 238],
                "LineWidth": 3.0,
                "ConstructionLineWidth": 2.0,
                "PointDiameter": 10.0,
                "UseDataDefinedColors": False
            }
        }),
        'aperture': StyleCollection.from_json({
            "FlowViewStyle": {
                "BackgroundColor": [30, 37, 43],
                "FineGridColor": [39, 49, 58],
                "CoarseGridColor": [17, 23, 28]
            },
            "NodeStyle": {
                "NormalBoundaryColor": [245, 193, 94],
                "SelectedBoundaryColor": [255, 209, 102],
                "GradientColor0": [247, 204, 115],
                "GradientColor1": [230, 162, 60],
                "GradientColor2": [191, 126, 37],
                "GradientColor3": [143, 91, 23],
                "ShadowColor": [10, 14, 18, 180],
                "FontColor": [251, 247, 241],
                "FontColorFaded": [220, 207, 186],
                "ConnectionPointColor": [245, 193, 94],
                "FilledConnectionPointColor": [143, 183, 201],
                "ErrorColor": [214, 95, 95],
                "WarningColor": [230, 162, 60],
                "PenWidth": 1.3,
                "HoveredPenWidth": 1.9,
                "ConnectionPointDiameter": 8.0,
                "Opacity": 0.98
            },
            "ConnectionStyle": {
                "ConstructionColor": [120, 138, 148],
                "NormalColor": [143, 183, 201],
                "SelectedColor": [255, 209, 102],
                "SelectedHaloColor": [255, 209, 102],
                "HoveredColor": [198, 227, 238],
                "LineWidth": 3.0,
                "ConstructionLineWidth": 2.0,
                "PointDiameter": 10.0,
                "UseDataDefinedColors": False
            }
        }),
        'screen': StyleCollection.from_json({
            "FlowViewStyle": {
                "BackgroundColor": [30, 37, 43],
                "FineGridColor": [39, 49, 58],
                "CoarseGridColor": [17, 23, 28]
            },
            "NodeStyle": {
                "NormalBoundaryColor": [224, 126, 126],
                "SelectedBoundaryColor": [255, 209, 102],
                "GradientColor0": [224, 138, 138],
                "GradientColor1": [214, 95, 95],
                "GradientColor2": [172, 70, 70],
                "GradientColor3": [124, 49, 49],
                "ShadowColor": [10, 14, 18, 180],
                "FontColor": [250, 244, 244],
                "FontColorFaded": [219, 194, 194],
                "ConnectionPointColor": [224, 126, 126],
                "FilledConnectionPointColor": [143, 183, 201],
                "ErrorColor": [214, 95, 95],
                "WarningColor": [230, 162, 60],
                "PenWidth": 1.3,
                "HoveredPenWidth": 1.9,
                "ConnectionPointDiameter": 8.0,
                "Opacity": 0.98
            },
            "ConnectionStyle": {
                "ConstructionColor": [120, 138, 148],
                "NormalColor": [143, 183, 201],
                "SelectedColor": [255, 209, 102],
                "SelectedHaloColor": [255, 209, 102],
                "HoveredColor": [198, 227, 238],
                "LineWidth": 3.0,
                "ConstructionLineWidth": 2.0,
                "PointDiameter": 10.0,
                "UseDataDefinedColors": False
            }
        })
    }
else:
    FLOW_SCENE_STYLE = None
    FLOW_NODE_STYLES = {}


if NodeData is not None:
    class _FlowEdgeData(NodeData):
        data_type = NodeDataType('beam-flow', 'Beam Flow')

        def __init__(self, payload=None):
            self.payload = payload
else:
    _FlowEdgeData = None


if NodeDataModel is not None:
    class _ElementNodeDataModel(NodeDataModel, verify=False):
        caption_visible = True

        def __init__(self, title, subtitle="", n_inputs=0, has_outputs=False,
                     element_id=None, style=None, node_kind='oe',
                     parent=None):
            super().__init__(style=style, parent=parent)
            self.name = "Beamline Element"
            self.caption = title
            self._subtitle = subtitle
            self._n_inputs = max(0, int(n_inputs))
            self._has_outputs = bool(has_outputs)
            self.element_id = element_id
            self.node_kind = node_kind
            self._widget = None

        @property
        def num_ports(self):
            return {
                PortType.input: self._n_inputs,
                PortType.output: 1 if self._has_outputs else 0,
            }

        @property
        def data_type(self):
            flow_type = _FlowEdgeData.data_type
            return {
                PortType.input: {
                    i: flow_type for i in range(self._n_inputs)},
                PortType.output: {
                    0: flow_type} if self._has_outputs else {},
            }

        @property
        def port_caption(self):
            return {
                PortType.input: {
                    i: "in {}".format(i+1) for i in range(self._n_inputs)},
                PortType.output: {
                    0: "out"} if self._has_outputs else {},
            }

        @property
        def port_caption_visible(self):
            return {
                PortType.input: {
                    i: self._n_inputs > 1 for i in range(self._n_inputs)},
                PortType.output: {
                    0: True} if self._has_outputs else {},
            }

        def set_in_data(self, node_data, port):
            return

        def out_data(self, port):
            return _FlowEdgeData()

        def embedded_widget(self):
            if self._widget is None:
                self._widget = qt.QFrame()
                self._widget.setFrameShape(qt.QFrame.NoFrame)
                layout = qt.QVBoxLayout(self._widget)
                layout.setContentsMargins(8, 2, 8, 6)
                if self._subtitle:
                    subtitle = qt.QLabel(self._subtitle)
                    subtitle.setWordWrap(True)
                    subtitle.setStyleSheet("color: palette(mid);")
                    layout.addWidget(subtitle)
            return self._widget
else:
    _ElementNodeDataModel = None


if ConnectionGraphicsObject is not None:
    class _FlowConnectionGraphicsObject(ConnectionGraphicsObject):
        def mousePressEvent(self, event):
            if (event.button() == qt.Qt.LeftButton and
                    self._connection.is_complete and
                    self._connection.required_port == PortType.none):
                node = self._connection.get_node(PortType.input)
                if node is not None:
                    interaction = NodeConnectionInteraction(
                        node, self._connection, self._scene)
                    interaction.disconnect(PortType.input)
                    event.accept()
                    return
            super().mousePressEvent(event)
else:
    _FlowConnectionGraphicsObject = None


if FlowScene is not None:
    class _ReadOnlyFlowScene(FlowScene):
        def create_node(self, data_model):
            node = Node(data_model)
            node.graphics_object = NodeGraphicsObject(self, node)
            self._nodes[node.id] = node
            self.node_created.emit(node)
            return node

        def create_connection(self, port_a, port_b=None, converter=None,
                              check_cycles=True):
            connection = super().create_connection(
                port_a, port_b=port_b, converter=converter,
                check_cycles=check_cycles)
            if _FlowConnectionGraphicsObject is not None:
                old_graphics = connection.graphics_object
                if old_graphics is not None:
                    old_graphics._cleanup()
                connection.graphics_object = _FlowConnectionGraphicsObject(
                    self, connection)
            return connection

        def _setup_connection_signals(self, conn):
            try:
                super()._setup_connection_signals(conn)
            except TypeError as exc:
                if 'not unique' not in str(exc).lower():
                    raise
else:
    _ReadOnlyFlowScene = None


class _FlowGraphPanel(qt.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if FlowScene is None:
            label = qt.QLabel(
                "Install `qtpynodeeditor` to enable the flow graph view.")
            label.setWordWrap(True)
            label.setAlignment(qt.Qt.AlignTop | qt.Qt.AlignLeft)
            layout.addWidget(label)
            layout.addStretch()
            self.scene = None
            self.view = None
            return

        scene_class = _ReadOnlyFlowScene if _ReadOnlyFlowScene is not None \
            else FlowScene
        self.scene = scene_class(
            style=FLOW_SCENE_STYLE,
            allow_node_creation=False, allow_node_deletion=False, parent=self)
        self.view = FlowView(self.scene, self)
        self.view.setRenderHint(qt.QPainter.Antialiasing, True)
        self.view.setContextMenuPolicy(qt.Qt.NoContextMenu)
        layout.addWidget(self.view)

    def set_graph(self, nodes, edges, order=None):
        if self.scene is None:
            return

        self.scene.clear_scene()
        order = list(order or nodes.keys())
        node_items = {}
        indegree = defaultdict(int)
        outgoing = defaultdict(list)

        for source_id, target_id in edges:
            if source_id in nodes and target_id in nodes:
                indegree[target_id] += 1
                outgoing[source_id].append(target_id)

        for node_id in order:
            node_info = nodes.get(node_id)
            if node_info is None:
                continue
            model = _ElementNodeDataModel(
                title=node_info['title'],
                subtitle=node_info.get('subtitle', ''),
                n_inputs=indegree.get(node_id, 0),
                has_outputs=bool(outgoing.get(node_id)),
                element_id=node_id,
                style=node_info.get('style'),
                node_kind=node_info.get('node_kind', 'oe'))
            node_items[node_id] = self.scene.create_node(model)

        input_index = defaultdict(int)
        for source_id, target_id in edges:
            source_node = node_items.get(source_id)
            target_node = node_items.get(target_id)
            if source_node is None or target_node is None:
                continue
            target_port = input_index[target_id]
            if target_port >= indegree[target_id]:
                continue
            self.scene.create_connection_by_index(
                target_node, target_port, source_node, 0, converter=None)
            input_index[target_id] += 1

        levels = {node_id: 0 for node_id in node_items}
        for _ in range(len(node_items)):
            changed = False
            for source_id, target_id in edges:
                if source_id not in levels or target_id not in levels:
                    continue
                next_level = levels[source_id] + 1
                if next_level > levels[target_id]:
                    levels[target_id] = next_level
                    changed = True
            if not changed:
                break

        rows_by_level = defaultdict(list)
        for node_id in order:
            if node_id in node_items:
                rows_by_level[levels.get(node_id, 0)].append(node_id)

        x_step = 280
        y_step = 140
        for level, node_ids in rows_by_level.items():
            for row, node_id in enumerate(node_ids):
                node_items[node_id].position = (level*x_step, row*y_step)

        if not node_items:
            return
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(
            -40, -40, 40, 40))
        self.view.fitInView(self.scene.sceneRect(), qt.Qt.KeepAspectRatio)
