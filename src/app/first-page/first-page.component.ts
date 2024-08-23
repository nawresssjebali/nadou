import { Component } from '@angular/core';
import { Router } from '@angular/router'; // Correct import


@Component({
  selector: 'app-first-page',
  standalone: true,
  imports: [],
  templateUrl: './first-page.component.html',
  styleUrl: './first-page.component.css'
})
export class FirstPageComponent {

  
    constructor(private router: Router) {}
  
    navigateToCourierDetails() {
      console.log('Navigating to CourierDetailsComponent...');
      this.router.navigate(['/with-courier']).then(success => {
        if (success) {
          console.log('Navigation successful');
        } else {
          console.log('Navigation failed');
        }
      }).catch(err => console.error('Navigation error:', err));
    }
  }
  

