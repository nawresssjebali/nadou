import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { NgModule } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-courier-details',
  templateUrl: './courier-details.component.html',
  standalone: true, // Add this line
  imports: [CommonModule, FormsModule,HttpClientModule],
  styleUrls: ['./courier-details.component.css']
})
export class CourierDetailsComponent {
  numberOfCouriers: number = 0;
  couriers: Array<any> = [];

  private apiUrl = 'http://localhost:5000/add_courier';

  constructor(private http: HttpClient) {}

  generateCourierLines() {
    this.couriers = [];
    for (let i = 0; i < this.numberOfCouriers; i++) {
      this.couriers.push({ area: '', price: 0, coordinateX: 0, coordinateY: 0 });
    }
  }

  onAddButtonClick(index: number) {
    const courierData = this.couriers[index];
  courierData.price = courierData.price;  // Ensure price is included
  this.http.post(this.apiUrl, courierData).subscribe(
    (response: any) => {
      console.log('Data processed successfully:', response);
    },
    (error) => {
      console.error('Error processing data:', error);
    }
  );
}

  submit() {
    console.log(this.couriers);
  }
}