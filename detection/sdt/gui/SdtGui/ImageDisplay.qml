// SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
//
// SPDX-License-Identifier: BSD-3-Clause

import QtQuick 2.0
import QtQuick.Controls 2.7
import QtQuick.Layouts 1.7
import SdtGui 0.1
import SdtGui.Templates 0.1 as T


T.ImageDisplay {
    id: root

    implicitHeight: viewButtonLayout.implicitHeight + contrastLayout.implicitHeight
    implicitWidth: contrastLayout.implicitWidth

    property list<Item> overlays

    onOverlaysChanged: {
        var cld = []
        cld.push(img)
        for (var i = 0; i < overlays.length; i++)
            cld.push(overlays[i])
        // Set contentChildren to re-parent items. Otherwise setting anchors
        // below will not work.
        scroll.contentChildren = cld

        for (var i = 0; i < overlays.length; i++) {
            var a = overlays[i]
            a.anchors.fill = img
            a.z = i
            // Check if scaleFactor property exists
            if (typeof a.scaleFactor !== undefined)
                a.scaleFactor = Qt.binding(function() { return scroll.scaleFactor })
        }
    }
    onImageChanged: {
        // First time an image is loaded automatically set contrast
        if (!rootLayout.imageLoaded) {
            contrastAutoButton.clicked()
            rootLayout.imageLoaded = true
        }
    }

    ColumnLayout {
        id: rootLayout
        anchors.fill: parent

        // Put here instead of in root to make them private
        property real contrastMin: 0.0
        property real contrastMax: 0.0
        property bool imageLoaded: false

        function calcScaleFactor(srcW, srcH, sclW, sclH) {
            var xf, yf
            if (srcW == 0) {
                xf = 1
            } else {
                xf = sclW / srcW
            }
            if (srcH == 0) {
                yf = 1
            } else {
                yf = sclH / srcH
            }
            return Math.min(xf, yf)
        }

        RowLayout{
            ColumnLayout {
                id: viewButtonLayout

                ToolButton {
                    id: zoomOutButton
                    icon.name: "zoom-out"
                    onClicked: {
                        scroll.scaleFactor /= Math.sqrt(2)
                        zoomFitButton.checked = false
                    }
                }
                ToolButton {
                    icon.name: "zoom-original"
                    onClicked: {
                        scroll.scaleFactor = 1.0
                        zoomFitButton.checked = false
                    }
                }
                ToolButton {
                    id: zoomFitButton
                    icon.name: "zoom-fit-best"
                    checkable: true
                }
                ToolButton {
                    icon.name: "zoom-in"
                    onClicked: {
                        scroll.scaleFactor *= Math.sqrt(2)
                        zoomFitButton.checked = false
                    }
                }
                Item {
                    Layout.fillHeight: true
                }
            }
            Item {
                Layout.fillWidth: true
                Layout.fillHeight: true

                ScrollView {
                    id: scroll

                    property real scaleFactor: 1.0
                    Binding on scaleFactor {
                        when: zoomFitButton.checked
                        value: rootLayout.calcScaleFactor(
                            img.sourceWidth, img.sourceHeight,
                            scroll.width, scroll.height
                        )
                    }

                    contentWidth: Math.max(availableWidth, img.width)
                    contentHeight: Math.max(availableHeight, img.height)
                    clip: true
                    anchors.fill: parent

                    PyImage {
                        id: img
                        anchors.centerIn: parent
                        source: root.image
                        black: rootLayout.contrastMin
                        white: rootLayout.contrastMax
                        width: sourceWidth * scroll.scaleFactor
                        height: sourceHeight * scroll.scaleFactor
                    }
                }
                Label {
                    text: "Error: " + root.error
                    visible: root.error
                    background: Rectangle {
                        color: "#50FF0000"
                        radius: 5
                    }
                    anchors.left: scroll.left
                    anchors.right: scroll.right
                    anchors.top: scroll.top
                    padding: 10
                }
            }
        }
        RowLayout {
            id: contrastLayout

            Label {
                text: "contrast"
            }
            RangeSlider {
                id: contrastSlider
                Layout.fillWidth: true

                from: root._imageMin
                to: root._imageMax
                stepSize: (to - from) / 100

                first.onMoved: { rootLayout.contrastMin = first.value }
                second.onMoved: { rootLayout.contrastMax = second.value }
            }
            Button  {
                id: contrastAutoButton
                text: "auto"
                onClicked: {
                    rootLayout.contrastMin = root._imageMin
                    contrastSlider.first.value = root._imageMin
                    rootLayout.contrastMax = root._imageMax
                    contrastSlider.second.value = root._imageMax
                }
            }
        }
    }

    Component.onCompleted: { zoomFitButton.checked = true }
}
