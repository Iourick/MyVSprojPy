// probe0.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <ctime>
#include <Windows.h>

// Function to get current date and time
std::string getCurrentDateTime() {
    time_t now = time(0);
    struct tm timeinfo;
    localtime_s(&timeinfo, &now);

    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
    return buffer;
}

int main() {
    // Get current date and time
    std::string currentDateTime = getCurrentDateTime();

    // Get computer name
    char computerName[MAX_COMPUTERNAME_LENGTH + 1];
    DWORD size = sizeof(computerName);
    GetComputerNameA(computerName, &size);

    // Name of the project
    std::string projectName = "YourProjectName"; // Replace this with your project's name

    // Prepare the content to write in the file
    std::string content = "Current Date: " + currentDateTime + "\n";
    content += "Computer Name: " + std::string(computerName) + "\n";
    content += "Project Name: " + projectName + "\n";

    // Write to a text file
    std::ofstream outputFile("info.log");
    if (outputFile.is_open()) {
        outputFile << content;
        outputFile.close();
        std::cout << "Information has been written to info.txt successfully." << std::endl;
    }
    else {
        std::cerr << "Unable to open the file for writing!" << std::endl;
    }

    return 0;
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
