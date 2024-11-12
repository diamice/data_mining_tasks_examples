import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QStackedWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class DBSCANApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DBSCAN Clustering App')

        # Layouts
        main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()

        # Поля ввода
        self.n_samples_input = QLineEdit(self)
        self.n_samples_input.setPlaceholderText("Количество точек (n_samples)")
        self.eps_input = QLineEdit(self)
        self.eps_input.setPlaceholderText("eps (радиус)")

        # Запуск кластеризации
        self.run_button = QPushButton("Сгенерировать и показать графики", self)
        self.run_button.clicked.connect(self.run_dbscan)

        # Виджет для смены для графиков
        self.stacked_widget = QStackedWidget()

        input_layout.addWidget(self.n_samples_input)
        input_layout.addWidget(self.eps_input)
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.run_button)
        main_layout.addWidget(self.stacked_widget)

        # Навигация в графиках
        self.prev_button = QPushButton("Назад", self)
        self.prev_button.clicked.connect(self.show_prev_plot)
        self.next_button = QPushButton("Вперед", self)
        self.next_button.clicked.connect(self.show_next_plot)

        main_layout.addWidget(self.prev_button)
        main_layout.addWidget(self.next_button)

        self.setLayout(main_layout)

        self.current_plot_index = 0
        self.num_plots = 3  # 3 Графика

    def run_dbscan(self):
        # Получение параметров из ввода пользователя
        try:
            n_samples = int(self.n_samples_input.text())
            eps = float(self.eps_input.text())
        except ValueError:
            print("Введите корректные значения")
            return

        # Генерация данных
        X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.60, random_state=0)

        # Удаление старых графиков
        while self.stacked_widget.count():
            widget = self.stacked_widget.widget(0)
            widget.deleteLater()
            self.stacked_widget.removeWidget(widget)

        # Шаг 1: Сгенерированные данные
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(X[:, 0], X[:, 1], label='True Position', color='blue')
        ax1.set_title("Шаг 1: Сгенерированные данные")
        ax1.set_xlabel('Признак 1')
        ax1.set_ylabel('Признак 2')
        ax1.legend()
        ax1.grid()

        # Добавляем первый график в StackedWidget
        canvas1 = FigureCanvas(fig1)
        self.stacked_widget.addWidget(canvas1)

        # Применение DBSCAN
        db = DBSCAN(eps=eps, min_samples=5).fit(X)
        labels = db.labels_

        # Шаг 2: Результаты кластеризации
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]  # Черный цвет для шума
            class_member_mask = (labels == k)

            # Выделяем основные точки (core samples)
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True

            xy = X[class_member_mask & core_samples_mask]
            ax2.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
                label=f'Кластер {k}' if k != -1 else 'Шум'
            )

            # Выделяем остальные точки (non-core samples)
            xy = X[class_member_mask & ~core_samples_mask]
            ax2.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        ax2.set_title("Шаг 2: Результаты кластеризации")
        ax2.set_xlabel('Признак 1')
        ax2.set_ylabel('Признак 2')
        ax2.grid()

        # Добавляем второй график в StackedWidget
        canvas2 = FigureCanvas(fig2)
        self.stacked_widget.addWidget(canvas2)

        # Шаг 3: Кластеры с центрами (если они есть)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
        ax3.set_title("Шаг 3: Кластеры")
        ax3.set_xlabel('Признак 1')
        ax3.set_ylabel('Признак 2')
        ax3.grid()

        # Добавляем третий график в StackedWidget
        canvas3 = FigureCanvas(fig3)
        self.stacked_widget.addWidget(canvas3)

        # Обновление текущего индекса и отображение первого графика
        self.current_plot_index = 0
        self.stacked_widget.setCurrentIndex(self.current_plot_index)

    def show_prev_plot(self):
        if self.current_plot_index > 0:
            self.current_plot_index -= 1
            self.stacked_widget.setCurrentIndex(self.current_plot_index)

    def show_next_plot(self):
        if self.current_plot_index < self.num_plots - 1:
            self.current_plot_index += 1
            self.stacked_widget.setCurrentIndex(self.current_plot_index)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DBSCANApp()
    ex.resize(800, 600)
    ex.show()
    sys.exit(app.exec_())
