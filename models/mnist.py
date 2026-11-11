import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self, channels, hidden_dim):
        super(MNISTModel, self).__init__()
        c1, c2 = channels
        self.feat = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        x = self.feat(x)
        x = self.classifier(x)
        return x

class TeacherCNN(MNISTModel):

    def __init__(self):
        super().__init__(
            channels=(64, 128),
            hidden_dim=256
        )


class StudentCNN(MNISTModel):

    def __init__(self):
        super().__init__(
            channels=(8, 16),
            hidden_dim=64
        )
class StudentCNNExs(MNISTModel):

    def __init__(self):
        super().__init__(
            channels=(4, 8),
            hidden_dim=32
        )
class StudentCNNExs2(MNISTModel):

    def __init__(self):
        super().__init__(
            channels=(2, 4),
            hidden_dim=16
        )
class TeacherCNNDropout(MNISTModel):

    def __init__(self, p=0.3):
        super().__init__(
            channels=(64, 128),
            hidden_dim=256
        )
        self.classifier.insert(3, nn.Dropout(p))


class StudentCNNDropout(MNISTModel):

    def __init__(self, p=0.3):
        super().__init__(
            channels=(8, 16),
            hidden_dim=64
        )
        self.classifier.insert(3, nn.Dropout(p))


class StudentCNNExsDropout(MNISTModel):

    def __init__(self, p=0.3):
        super().__init__(
            channels=(4, 8),
            hidden_dim=32
        )
        self.classifier.insert(3, nn.Dropout(p))


class StudentCNNExs2Dropout(MNISTModel):

    def __init__(self, p=0.3):
        super().__init__(
            channels=(2, 4),
            hidden_dim=16
        )
        self.classifier.insert(3, nn.Dropout(p))

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    teacher = TeacherCNN()
    student = StudentCNN()

    teacher.eval()
    student.eval()

    x = torch.randn(4, 1, 28, 28)

    with torch.no_grad():
        teacher_output = teacher(x)
        student_output = student(x)

    print("Input shape:", x.shape)

    print("Teacher output:", teacher_output.shape)
    print("Student output:", student_output.shape)

    print("Teacher parameters:", count_parameters(teacher))
    print("Student parameters:", count_parameters(student))

    assert teacher_output.shape == (4, 10)
    assert student_output.shape == (4, 10)

    assert torch.isfinite(teacher_output).all()
    assert torch.isfinite(student_output).all()

    print("Model smoke test passed!")
