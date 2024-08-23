import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-schema',
  standalone: true,
  imports: [CommonModule, HttpClientModule],
  templateUrl: './schema.component.html',
  styleUrls: ['./schema.component.css']
})
export class SchemaComponent implements OnInit {
  isLoading = true; // Initially true to show loading message
  imageUrl?: string; // Optional property to hold the image URL

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.computeRouteAndFetchImage();
  }

  computeRouteAndFetchImage() {
    // First, trigger the POST request to compute the route
    this.http.post('http://localhost:5000/compute-route_1', {}, { responseType: 'text' })
  .subscribe({
    next: (response: any) => {
      console.log('Route computed successfully:', response);
      this.getGeneratedImage();
    },
    error: (error) => {
      console.error('Error computing route:', error);
      this.isLoading = false;
    }
  });
  }

  getGeneratedImage() {
    // Fetch the image after the route is computed
    this.imageUrl = 'http://localhost:5000/get-plot';
    this.isLoading = false; // Set to false to hide the loading message
  }
}






