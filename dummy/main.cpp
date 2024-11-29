#include "gui.h"
#include <QApplication>

int main(int argc, char **argv) {
    QApplication app(argc, argv);
    GUI gui;
    gui.show();
    return app.exec();
}
