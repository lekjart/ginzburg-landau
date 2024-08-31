#include <stdio.h>
#include <windows.h>
#include <GL/glew.h>
#include <GL/freeglut.h>


#include <helper_gl.h>
#include "resource.h"
#include "Solver.h"


int lastx, lasty;
bool clicked = false;
GLuint imageBuf;
HWND myDialog;
void *pixelBuffer;
int saveFile = 0;

extern void InitSolver(int, void **);
extern void ReleaseSolver(void *);
extern bool DoStuff(float *, int, int, void *);
extern void UpdateCudaParas(void);
extern void ResetField(void);
extern void UpdatePalette(const COLORREF *);
extern void CreateZero(unsigned int, unsigned int);

extern float h_Para[];

struct BezierPoint {
	COLORREF anchor[4];
};

BezierPoint paletteList[100];
int currentPalette = -1;
int currentData = 0;

void reshape(int x, int y)
{
	return;
}

void motion(int x, int y)
{
	return;
}

void click(int button, int updown, int x, int y)
{
	lastx = x;
	lasty = y;
	clicked = !clicked;

	if (updown) {
		//CreateZero(x , y);
		saveFile = 1;
	}
}

void keyboard(unsigned char key, int x, int y)
{
	return;
}

void display(void)
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imageBuf);
	glDrawPixels(FIELDSIZEX, FIELDSIZEY, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glFlush();
}

void SaveBitmap(char *path, void *buf)
{
	FILE *Out = fopen(path, "wb");
	if (!Out)
		return;
	BITMAPFILEHEADER bitmapFileHeader;
	BITMAPINFOHEADER bitmapInfoHeader;

	bitmapFileHeader.bfType = 0x4D42;
	bitmapFileHeader.bfSize = FIELDSIZEX*FIELDSIZEY * 3;
	bitmapFileHeader.bfReserved1 = 0;
	bitmapFileHeader.bfReserved2 = 0;
	bitmapFileHeader.bfOffBits =
		sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapInfoHeader.biWidth = FIELDSIZEX - 1;
	bitmapInfoHeader.biHeight = FIELDSIZEY - 1;
	bitmapInfoHeader.biPlanes = 1;
	bitmapInfoHeader.biBitCount = 24;
	bitmapInfoHeader.biCompression = BI_RGB;
	bitmapInfoHeader.biSizeImage = 0;
	bitmapInfoHeader.biXPelsPerMeter = 0; // ?
	bitmapInfoHeader.biYPelsPerMeter = 0; // ?
	bitmapInfoHeader.biClrUsed = 0;
	bitmapInfoHeader.biClrImportant = 0;

	fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, Out);
	fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, Out);
	fwrite(buf, FIELDSIZEX*FIELDSIZEY * 3, 1, Out);
	fclose(Out);
}

void idle(void)
{
	float elapsedTime;
	static char text[1024];
	if (DoStuff(&elapsedTime, currentData, 10, pixelBuffer))
	{
		if (saveFile) {
			saveFile = 0;
			SaveBitmap( "tmp.bmp", pixelBuffer);
		}
		sprintf_s(text, "Forced Ginzburg-Landau Equation %7.3f ms", elapsedTime);
		glutSetWindowTitle(text);
		glutPostRedisplay();

	}

}

void LoadPalettes(void)
{
	FILE *fp;
	char paletteName[1024];
	//GetCurrentDirectory(1024, paletteName);

	currentPalette = -1;
	fopen_s(&fp, "palettes.txt", "r");

	if (!fp)
		return;

	// Clear the listbox
	SendDlgItemMessage(myDialog, IDC_LIST1, LB_RESETCONTENT, 0, 0);

	int c[12];
	int idx;
	// Read in each palette entry
	while (fscanf_s(fp, "%s%d%d%d%d%d%d%d%d%d%d%d%d", paletteName, 1024, c, c+1, c+2, c+3, c+4, c+5, c+6, c+7, c+8,c+9,c+10,c+11) != EOF) {
		
		idx = (int)SendDlgItemMessage(myDialog, IDC_LIST1, LB_ADDSTRING, 0, (LPARAM)paletteName);
		paletteList[idx].anchor[0] = RGB(c[0], c[1], c[2]);
		paletteList[idx].anchor[1] = RGB(c[3], c[4], c[5]);
		paletteList[idx].anchor[2] = RGB(c[6], c[7], c[8]);
		paletteList[idx].anchor[3] = RGB(c[9], c[10], c[11]);

		printf("Read %s %d %d %d\n", paletteName, c[9], c[10], c[11]);
	}

	SendDlgItemMessage(myDialog, IDC_LIST1, LB_SETCURSEL, idx, 0);
	currentPalette = idx;
	UpdatePalette(paletteList[idx].anchor);

	fclose(fp);
}

INT_PTR CALLBACK MyDialogProc(
	_In_ HWND   hwndDlg,
	_In_ UINT   uMsg,
	_In_ WPARAM wParam,
	_In_ LPARAM lParam
)
{
	switch (uMsg)
	{
	case WM_COMMAND:
	{
		int controlID = LOWORD(wParam);
		int msg = HIWORD(wParam);
		// Check if this one of the parameter edit controls
		if (controlID >= IDC_EDIT1 && controlID < IDC_EDIT1 + PARACOUNT) {
			// map to which parameter
			int paraID = controlID - IDC_EDIT1;
			if (HIWORD(wParam) == EN_CHANGE) {
				char szText[128];
				GetDlgItemText(hwndDlg, controlID, (LPSTR)szText, 128);
				float val = (float)atof(szText);
				if (val != h_Para[paraID]) {
					// Values has changed, update it
					h_Para[paraID] = val;
					//printf("Would update %d to %f\n", paraID, val);
					UpdateCudaParas();
				}

			}
		} // end if parameter edits
		else if (controlID == IDC_BUTTON1) {
			if (msg == BN_CLICKED) {
				LoadPalettes();
			}
		} // end if palette reload
		else if (controlID == IDC_BUTTON2) {
			if (msg == BN_CLICKED) {
				ResetField();
			}
		} // end if reset field
		else if (controlID == IDC_LIST1) {
			if (msg == LBN_SELCHANGE) {
				int selected = (int)SendDlgItemMessage(myDialog, IDC_LIST1, LB_GETCURSEL, 0, 0);
				if (selected == LB_ERR)
					break;

				COLORREF color = paletteList[selected].anchor[0];
				printf("Item selected with data %d %d %d\n", GetRValue(color), GetGValue(color), GetBValue(color));
				currentPalette = selected;
				UpdatePalette(paletteList[selected].anchor);
			}
		} // end if palette selectino
		else if (controlID == IDC_COMBO1) {
			if (msg == CBN_SELENDOK) {
				int selected = (int)SendDlgItemMessage(myDialog, IDC_COMBO1, CB_GETCURSEL, 0, 0);
				printf("Selected data %d\n", selected);
				currentData = selected;
			}

		} // end of data selector
	}

		break;
	default:
		return FALSE;
	}

	return FALSE;
}

void InitParas(void)
{
	for (int i = 0; i < PARACOUNT; i++) {
		char text[128];
		sprintf_s(text, "%7.3f", h_Para[i]);
		SetDlgItemText(myDialog, IDC_EDIT1 + i, (LPSTR)text);
	}
}



int main(int argc, char **argv)
{
	// Initialize GL windows
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(FIELDSIZEX, FIELDSIZEY);
	glutCreateWindow("Forced Ginzburg-Landau Equation");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(click);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);
	glEnable(GL_TEXTURE_2D);
	glewInit();

	glGenBuffers(1, &imageBuf);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imageBuf);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, FIELDSIZEX*FIELDSIZEY * 4, NULL, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	
	InitSolver(imageBuf, &pixelBuffer);

	// Start a windows dialog
	myDialog = CreateDialog(NULL, MAKEINTRESOURCE(IDD_DIALOG1), 0, MyDialogProc);

	// Populate drawing options
	SendDlgItemMessage(myDialog, IDC_COMBO1, CB_ADDSTRING, 0, (LPARAM)"Modulus");
	SendDlgItemMessage(myDialog, IDC_COMBO1, CB_ADDSTRING, 0, (LPARAM)"Phase");
	SendDlgItemMessage(myDialog, IDC_COMBO1, CB_ADDSTRING, 0, (LPARAM)"U");
	SendDlgItemMessage(myDialog, IDC_COMBO1, CB_ADDSTRING, 0, (LPARAM)"V");
	SendDlgItemMessage(myDialog, IDC_COMBO1, CB_SETCURSEL, 0, 0);
	//SendDlgItemMessage(myDialog, IDC_COMBO1, CB_SETEXTENDEDUI, 1, 0);

	ShowWindow(myDialog, SW_SHOW);
	LoadPalettes();
	InitParas();
	
	glutMainLoop();

	ReleaseSolver(pixelBuffer);

	glDeleteBuffers(1, &imageBuf);

	printf("Done\n");
	return 0;
}

